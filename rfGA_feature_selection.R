normalize <- function(x) { # z-score normalization
  return ((x-mean(x))/sd(x))
}

# Set datasets directory   
setwd("C:/Users/lenovo/Downloads/Datasets/CI/hasil run TA CI")
# Read dataset file
mice <- read.csv("Data_Cortex_Nuclear.csv")

### Begin - PRAPROSES ####
mice_clean <- mice[,-c(1,79,80,81)] # remove mouseID, genotype, treatment, and behaviour attributes

# input missing values with attribute's mean with respect to its corresponding class
for(i in seq(ncol(mice_clean)-1)){
  mice_clean[,i] <- ave(mice_clean[,i], mice_clean$class, FUN=function(x) replace(x, is.na(x), mean(x, na.rm=TRUE)))
}
# normalize using z-score
mice_norm <- as.data.frame(lapply(mice_clean[1:(ncol(mice_clean)-1)], normalize))
mice_norm$class <- mice_clean$class
### End - PRAPROSES ####

#### Begin - PEMBAGIAN DATA ####
#untuk pembagian data
class_pointer <- c(150,150,135,135,135,135,105,135) # jumlah @ c-CS-m, c-SC-m, c-CS-s, c-SC-s, t-CS-m, t-SC-m, t-CS-s, t-SC-s
subset_data <- list()
for(j in seq(8)){ 
  if(j == 1) class_from.to <- c(j:sum(class_pointer[1:j])) 
  else class_from.to <- c((sum(class_pointer[1:(j-1)])+1):sum(class_pointer[1:j])) 
  subset_data <- c(subset_data, list(class_from.to))
}

# 1 - normal learning (c-CS-m, c-SC-m, c-CS-s, c-SC-s)
first_mice <- rbind(mice_norm[subset_data[[1]],], mice_norm[subset_data[[2]],], mice_norm[subset_data[[3]],], mice_norm[subset_data[[4]],])
# 2 - rescued learning - Case mice (Context-Shock -- t-CS-m, t-CS-s)
second_mice <- rbind(mice_norm[subset_data[[5]],], mice_norm[subset_data[[7]],])
# 3 failed learning - case vs control (c-CS-m, c-CS-s, t-CS-s)
third_mice <- rbind(mice_norm[subset_data[[1]],], mice_norm[subset_data[[3]],], mice_norm[subset_data[[7]],])
#### End - PEMBAGIAN DATA ####

library(caret)
library(desirability)
library(doParallel)

# banyaknya iterasi (generasi)
n.iter = 100
# ukuran populasi
pop.size = 20
# banyak core CPU yang digunakan
n.cores <- 4
# pengaturan random forest
rfGA2 <- rfGA
rfGA2$fitness_intern <- function (object, x, y, maximize, p) {
  Accuracy <- rfStats(object)[1]
  d_Accuracy <- desirability::dMax(0.75, 1)
  d_Size <- desirability::dMin(1, p, 2)
  overall <- desirability::dOverall(d_Accuracy, d_Size)
  D <- predict(overall, data.frame(Accuracy, ncol(x)))
  c(D = D, Accuracy = as.vector(Accuracy))
}

# pengaturan genetic algorithm
ga_ctrl_d <- gafsControl(functions = rfGA2,
                         method = "cv",
                         number = 5,
                         metric = c(internal = "D", external = "Accuracy"),
                         maximize = c(internal = TRUE, external = TRUE),
                         genParallel = TRUE,
                         allowParallel = TRUE,
                         verbose = TRUE)

for(k in seq(3)){
  if(k == 1) dataset <- first_mice
  else if(k == 2) dataset <- second_mice
  else dataset <- third_mice
  
  protein <- dataset[,-c(ncol(dataset))]
  class <- as.factor(as.character(dataset[,ncol(dataset)]))
  
  print(paste0("Running rfGA feature selection with dataset-",k," (",n.iter," iter | ", pop.size," pop size)"))
  start <- Sys.time()
  
  set.seed(10)
  cl <- makeCluster(n.cores)
  registerDoParallel(cl, cores = n.cores)
  rf_ga_d <- gafs(x = protein, y = class,
                  popSize = pop.size, iters = n.iter,
                  gafsControl = ga_ctrl_d)
  stopCluster(cl)
  
  end <- Sys.time()
  print(end - start)
  
  save.image(paste0("TA.CI_",pop.size,"_",n.iter,"iter_data",k,".RData"))
  remove(rf_ga_d,dataset)
}