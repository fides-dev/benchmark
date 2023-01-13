This repository contains the supplementary information accompanying the
manuscript "Fides: Reliable Trust-Region Optimization for Parameter Estimation
of Ordinary Differential Equation Models". 

To set up the scripts, execute the script `setup.sh`, this will setup a
 virtual environment and download and install additional dependencies.
 
To ensure reproducibility, the scripts are organized using snakemake. To
 execute the python part of the benchmark, execute `benchmarkLocal.sh
 `. This will run the whole benchmark locally and likely need multiple days
  to finish. However, it should be easy to adapt this script to run on a
   cluster, which can finish in a couple of hours depending on available
    resources.
    
Running the python benchmark will generate results files in the `./results
` directory that include optimization results in hdf5 format as well as
 waterfall, parameter and convergence plots for every model and otimizer
 . Moreover additional evaluation figures will be generated in the
  `./evaluation` directory.
    
The MATLAB part of this benchmark can be tun by executing the script
 `./Hass2019/run_Benchmark.m`. The script will take a bit over a week to
  finish on modern hardware. The folder `./Hass2019` contains a modified
   version of the script `arFit.m` that ensures that all optimizer options
    are corrrectly applied to the fmincon optimizer. These changes will be
     automatically applied to the downloaded d2d version at `./Hass2019/d2d
     `. This repository is already prepopulated with results from this
      optimization, which will be loaded instead of rerunning the benchmarks
      . To rerun a benchmark delete the respective `.mat` file in `./Hass2019`
      
To compare results across optimizers and methods, the script `comparison.py
` has to be executed after both MATLAB and python part of the benchmark have
 finished. This will generate additional figures and `.csv` files which
  serve as the basis for text and figure in the manuscript.
     


