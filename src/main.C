//  SW4 LICENSE
// # ----------------------------------------------------------------------
// # SW4 - Seismic Waves, 4th order
// # ----------------------------------------------------------------------
// # Copyright (c) 2013, Lawrence Livermore National Security, LLC.
// # Produced at the Lawrence Livermore National Laboratory.
// #
// # Written by:
// # N. Anders Petersson (petersson1@llnl.gov)
// # Bjorn Sjogreen      (sjogreen2@llnl.gov)
// #
// # LLNL-CODE-643337
// #
// # All rights reserved.
// #
// # This file is part of SW4, Version: 1.0
// #
// # Please also read LICENCE.txt, which contains "Our Notice and GNU General Public License"
// #
// # This program is free software; you can redistribute it and/or modify
// # it under the terms of the GNU General Public License (as published by
// # the Free Software Foundation) version 2, dated June 1991.
// #
// # This program is distributed in the hope that it will be useful, but
// # WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// # conditions of the GNU General Public License for more details.
// #
// # You should have received a copy of the GNU General Public License
// # along with this program; if not, write to the Free Software
// # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA
//#include "mpi.h"

#include "EW.h"

#include <cstring>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <mpi.h>
#ifndef SW4_NOOMP
#include <omp.h>
#endif
#include "version.h"

using namespace std;

// GLOBAL VARS
std::unique_ptr<EW> simulation;

void usage(string thereason)
{
  cout << endl
       << "sw4 - Summation by parts 4th order forward seismic wave propagator"  << endl << endl
       << "Usage: sw4 [-v] file.in" << endl
       << "\t -v:      prints out the version info" << endl
       << "\t file.in: an input file" << endl << endl
       << "Reason for message: " << thereason << endl;
}

void s_handles(MPIX_Handles * handle)
{
	std::vector<Image *> v = simulation->get_image_vector();
	// Each image has three communicators (2 from it's P_IO, and one from itself)
	// the "two" per P_IO is a hardcoded value right now. in the future,
	// probably need to get exact number.
	handle->comm_size = v.size()*3 + 1 ;
	handle->comms = new MPI_Comm[handle->comm_size];

	// Save Image objects' communicator
	for(int i = 0; i < v.size(); i++)
	{
		handle->comms[(i*3)] = v[i]->get_mpi_comm();
		// The Parallel_IO object with each image
		Parallel_IO* p_io = v[i]->get_pio();
		handle->comms[(i*3)+1] = p_io->get_write_comm();
		handle->comms[(i*3)+2] = p_io->get_data_comm();
	}

	// Save Cartesian communicator -- inside EW object
	handle->comms[v.size()*3] = simulation->m_cartesian_communicator;

	// SW4 doesn't need any extra groups to be saved (the comm groups are auto saved)
	handle->grps = nullptr;
	handle->group_size = 0;

	// Save the datatypes!
	std::vector<MPI_Datatype> lots_o_types = simulation->get_all_datatypes();
	handle->type_size = lots_o_types.size();
	handle->dtypes = new MPI_Datatype[handle->type_size];
	for(int i = 0; i < lots_o_types.size(); i++)
	{
		handle->dtypes[i] = lots_o_types[i];
	}

}

void d_handles(MPIX_Handles handle)
{
	// Restore the comms!
	for(int i = 0; i < handle.comm_size; i++)
	{
		//if(i < )
		//else if (i <)
		//else if (i < )
		//else
			// Restore the m_cartesian_communicator;
	}

	// Restore the types!
	simulation->restore_types(handle.dtypes, handle.type_size);
	delete[] handle.dtypes;
	delete[] handle.comms;
	delete[] handle.grps;
}

int main_loop(int fault_epoch, int *done, int myRank, int nProcs, const string& fileName);
int main(int argc, char **argv)
{
	int myRank = 0, nProcs = 0;
	string fileName;
	bool checkmode = false;

	stringstream reason;
	std::cout << "Starting up!" << std::endl;
	// Initialize MPI...
	MPI_Init(&argc, &argv);
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

	MPIX_Serialize_handler_register(s_handles);
	MPIX_Deserialize_handler_register(d_handles);

#ifdef ENABLE_TAU
   TAU_PROFILE_INIT(argc, argv);
#endif

  int status = 0;

  // mpi2 adds on four more args...  [-p4pg, dir, -p4wd, dir]
  int mpi2args = 4;

	if (argc != 2 && argc != 3 && argc != (2+mpi2args) && argc != (3+mpi2args) )
	{
		reason << "Wrong number of args (1-2), not: " << argc-1 << endl;

		if (myRank == 0)
		{
			for (int i = 0; i < argc; ++i)
			  cout << "Argv[" << i << "] = " << argv[i] << endl;

			usage(reason.str());
		}

		// Stop MPI
		MPI_Finalize();
		return 1;
	}
  else if (strcmp(argv[1],"-v") == 0 )
  {
     if (myRank == 0)
        cout << ewversion::getVersionInfo() << endl;
// Stop MPI
     MPI_Finalize();
     return status;
  }
  else
     if (argc == 1)
     {
        reason  << "ERROR: ****No input file specified!" << endl;
        for (int i = 0; i < argc; ++i)
           reason << "Argv[" << i << "] = " << argv[i] << endl;

        if (myRank == 0) usage(reason.str());
// Stop MPI
	MPI_Finalize();
        return 1;
     }

  else
    fileName = argv[1];

  if (myRank == 0)
  {
    cout << ewversion::getVersionInfo() << endl;
    cout << "Input file: " << fileName << endl;
  }

  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);

  cout.precision(8);
// use sci format: 1.2345e-6
  cout << std::scientific;
	int done = 0;
	int fault_epoch = 0;
	int code = MPI_SUCCESS;
	while(0 == done)
	{
		if(MPIX_TRY_RELOAD == code)
		{
			std::cout << "Freeing old resources!" << std::endl;
			// Looks like another process has failed
			// Free old resources
			MPIX_Deallocate_stale_resources(s_handles);
			// Restore MPI Environment
			MPIX_Checkpoint_read();
		}
		else if(MPI_SUCCESS != code)
		{
			cout  << "============================================================" << endl
			<< "The execution on proc " << myRank << " was UNSUCCESSFUL." << endl
			<< "============================================================" << endl;
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		MPIX_Get_fault_epoch(&fault_epoch);
		std::cout << "Epoch : " << fault_epoch << std::endl;
		code = main_loop(fault_epoch, &done, myRank, nProcs, fileName);
	}

	// Stop MPI
	MPI_Finalize();

	if( myRank == 0 )
	{
		cout << "============================================================" << endl
				 << " program sw4 finished! " << endl
				 << "============================================================" << endl;
	}

	std::cout << status << std::endl;
	return status;
} // end of main

int main_loop(int fault_epoch, int *done, int myRank, int nProcs, const string& fileName)
{
	int fail_step = 17;
	if(fault_epoch > 0)
		fail_step = -1;
// Save the source description here
  vector<vector<Source*> > GlobalSources;
// Save the time series here
  vector<vector<TimeSeries*> > GlobalTimeSeries;

// make a new simulation object by reading the input file 'fileName'
  simulation = std::make_unique<EW>(fileName, GlobalSources, GlobalTimeSeries, fault_epoch);
	if(fault_epoch > 0)
	{
		MPIX_Deserialize_handles();
		std::cout << "Z: " <<   fault_epoch << std::endl;
		//setupMPICommunications();
	}

  if (!simulation->wasParsingSuccessful())
  {
    if (myRank == 0)
    {
      cout << "Error: there were problems parsing the input file" << endl;
    }
    return -1; // Not sure on best value to use here. Don't want MPI_SUCCESS or MPIX_TRY_RELOAD
  }
  else
  {
		// get the simulation object ready for time-stepping
		simulation->setupRun( GlobalSources );
		std::cout << "After setup run " << std::endl;

		if (!simulation->isInitialized())
		{
			if (myRank == 0)
			{
				cout << "Error: simulation object not ready for time stepping" << endl;
			}
			return -1;
		}
		else
		{
			if (myRank == 0)
			{
				int nth=1;
				#ifndef SW4_NOOMP
				#pragma omp parallel
				{
					if( omp_get_thread_num() == 0 )
					{
						nth=omp_get_num_threads();
					}
				}
				#endif
				if( nth == 1 )
				{
				  if( nProcs > 1 )
				     cout << "Running sw4 on " <<  nProcs << " processors..." << endl;
				  else
				     cout << "Running sw4 on " <<  nProcs << " processor..." << endl;
				}
				else
				{
				  if( nProcs > 1 )
				     // Assume same number of threads for each MPI-task.
				     cout << "Running sw4 on " <<  nProcs << " processors, using " << nth << " threads/processor..." << endl;
				  else
				     cout << "Running sw4 on " <<  nProcs << " processor, using " << nth << " threads..." << endl;
				}
				//FTNC	 if( simulation->m_croutines )
				//FTNC	    cout << "   Using C routines." << endl;
				//FTNC	 else
				//FTNC	    cout << "   Using fortran routines." << endl;
				cout << "Writing output to directory: " << simulation->getPath() << endl;
			}
// run the simulation
      int ng=simulation->mNumberOfGrids;
      vector<DataPatches*> upred_saved(ng), ucorr_saved(ng);
      vector<Sarray> U(ng), Um(ng), ph(ng);
      simulation->solve( GlobalSources[0], GlobalTimeSeries[0], simulation->mMu,
			simulation->mLambda, simulation->mRho, U, Um, upred_saved,
			ucorr_saved, false, 0, 0, 0, ph, fail_step );

			int bob;
			MPIX_FT_errno(&bob);
			if(bob == MPIX_TRY_RELOAD)
			{
				std::cout << " hmmm error!" << std::endl;
				return bob;
			}
			std::cout << " Past check!" << std::endl;
// save all time series

      double myWriteTime = 0.0, allWriteTime;
      for (int ts=0; ts<GlobalTimeSeries[0].size(); ts++)
      {
				GlobalTimeSeries[0][ts]->writeFile();
				#ifdef USE_HDF5
					myWriteTime += GlobalTimeSeries[0][ts]->getWriteTime();
					if( ts == GlobalTimeSeries[0].size()-1)
					{
						GlobalTimeSeries[0][ts]->closeHDF5File();

						MPI_Reduce(&myWriteTime, &allWriteTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
						if( myRank == 0 )
							cout << "  ==> Max wallclock time to write time-series data is " << allWriteTime << " seconds." << endl;
					}
				#endif
			}

			*done = 1;
    }
  }
	return MPI_SUCCESS;
} // end of main
