## library
library(data.table)
library(optimx)
library(pso)

## Global parameters

# folder to the competition code
competition_folder = "c:/gitRepo/SE/cps-tool-competition"
## set wd
setwd(competition_folder)

# time budget for a run (in 2022 total time was 2h)
time_budget = 3600
# number of replicas
n_replica <- 5
# oob tollerance (Value used in 2022 competition)
oob_tol <- 0.85
# competition script
run_script = paste0("python competition.py --time-budget ",
                    time_budget, 
                    " --oob-tolerance ",oob_tol,
                    " --executor beamng  --map-size 200   --beamng-user C:\\tmp\\BeamNG --beamng-home=c:\\LocalWorks\\App\\BeamNG.tech.v0.26.2.0 ")
module_and_class = "--module-name mbt_generator.MBTGenerator --class-name MBTGenerator"
run_str = paste0(run_script,module_and_class)


resuls_folder = paste0(competition_folder,"/results/")

# EvoMBT parameter file
par_file = "mbt_generator/params.csv"
# execution report file
report_file = "mbt_generator/report.csv"
# basic parameters for EvoMBT
basic_par_file = "mbt_generator/Rscripts/basicParams.csv"
basic_par = fread(basic_par_file, data.table = F)
colnames(basic_par) <- c("var","value")



# cps parameters
# not really a good implementation

## Function to optimize is the number of oob. The function call competition.py and read the number of oob
##beamng_n_directions <- 16
##beamng_max_angle <- 80
#par <- c(8, 45,20,28,5)
compute_oob <- function(par){
  
  ## set wd
  setwd(competition_folder)
  
  # create parameter file adding par to basic parameters
  this_par = basic_par
  for( i in seq(1,length(par),1)){
    this_par[nrow(this_par)+1,] <- c(par_names[i], par[i])
  }
  fwrite(this_par,par_file, sep = ",", quote = F, append = F, row.names = F, col.names = F )
  
  # do for n replica and record n obb and n valid tests
  oobs <- c()
  valids <- c()
  for(i in seq(1,n_replica,1)){
    # clear previous test dir
    unlink(list.dirs(resuls_folder, recursive = F), recursive = T)
    
    # run generation
    shell(run_str, wait = T, translate = T, mustWork = F)  
    
    # read oob
    test_dir <- list.dirs(resuls_folder, recursive = F)
    oob_file = paste0(test_dir,"/oob_stats.csv")
    oob_data = fread(oob_file)
    n_oob <- oob_data$total_oob
    stats_file = paste0(test_dir,"/generation_stats.csv")
    stats_data = fread(stats_file)
    n_valid <- stats_data$test_valid
    
    
    ## write report
    report <- data.frame(matrix(ncol = length(par_names)+1, nrow = 0))
    colnames(report) <- c(par_names, "oob")
    for( i in seq(1,length(par),1)){
      report[1,par_names[i]] <-par[i]
    }
    report[1,"oob"] <- n_oob
    #report <- cbind(report,stats_data)
    report <- merge(report,stats_data)
    report$time_budget <- time_budget
    fwrite(report, report_file, quote = F, append = T, sep = ",", row.names = F, col.names = F )
    
    oobs <- c(oobs, n_oob)
    valids <- c(valids, n_valid)
  }
  
  # take the median of oob and n valid
  n_oob <- median(oobs)
  n_valid <- median(valids)
  
  # weight of n obb 
  penalty_prop <- 0.95
  penalty <- n_oob * penalty_prop +  n_valid * ( 1 - penalty_prop )
  
  # revert to minimize
  return(-1 * penalty)
}


## define initial parameters and boundaries
par_names = c("Dbeamng_n_directions","Dbeamng_max_angle",
              "Dbeamng_min_street_length","Dbeamng_max_street_length","Dbeamng_street_chunck_length")
initial_par <- c(24, 45,15,30,15)
min_par <- c(8, 15, 10, 20, 5)
max_par <- c(45, 60, 20, 40, 15)
names(initial_par) <- par_names

## Create report empty report
## Use this to erease old report
report <- data.frame(matrix(ncol = length(par_names)+11, nrow = 0))
colnames(report) <- c(par_names, c("oob",
                                   "test_generated", "test_valid", "test_invalid", "test_passed", "test_failed", "test_in_error", 
                                   "real_time_generation", "real_time_execution", "simulated_time_execution","time_budget"))
fwrite(report, report_file, quote = F, append = F, sep = ",", row.names = F, col.names = T )

#######################
# Optimizations
# Evaluate the time required

# Uncomment to apply gradient based optimization
# opt <- optimx(par = initial_par, fn = compute_oob, method = "L-BFGS-B",lower = min_par, upper = max_par, itnmax = 5,
#              control = list(trace = 0, all.methods = FALSE,
#                             ndeps = rep(1, length(initial_par))))

# Uncomment to apply PSO 
# opt <- psoptim(initial_par, fn = compute_oob, lower = min_par, upper = max_par, 
#         control = list(maxit = 10, s = 10, w = 0.5, c.p = 0.4, c.g = 0.6))


#######################
# Search on given values

search_space <- data.frame(matrix(ncol = length(par_names), nrow = 0))
colnames(search_space) <- par_names
search_space[1,] <- c(8,45,10,30,10)
search_space[2,] <- c(12,60,20,30,10)
search_space[3,] <- c(24,45,25,35,10)
search_space[4,] <- c(36,50,20,30,10)

# i<-1
for( i in seq(1,nrow(search_space),1)){
  cat("#################### ",i,"\n")
  compute_oob(search_space[i,])
}
