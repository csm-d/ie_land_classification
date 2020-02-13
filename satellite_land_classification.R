library(sp)
library(raster)
library(mapview)
library(keras)  

#LOAD RAW DATA
#Loading Landsat8 Data
SR_LC8_files <- list.files("data/landsat/SR/", pattern = "^LC8.*(2018|2019|2020).*_G07.dat$", full.names = TRUE)
BT_LC8_files <- list.files("data/landsat/BT/", pattern = "^LC8.*(2018|2019|2020).*_G07.dat$", full.names = TRUE)

#Replacing NAs with meaningful data in the latest snapshot 
#dimensions: 1000, 975, 975000, (2,7)  (nrow, ncol, ncell, nlayers)
#resolution : 30, 30  (x, y)
brick_files <- function(files)
{
  last_stack <- stack(files[length(files)])
  tmp_names <- names(last_stack)
  for (i in (length(files)-1):1) {
    cat('Processing layer', i, 'of', length(files),'\n')
    for (k in 1:nlayers(last_stack)) {
      last_stack[[k]] <- cover(last_stack[[k]], stack(files[i])[[k]])
    }
  }
  names(last_stack) <- tmp_names
  result <- brick(last_stack)
}

sr_brick <- brick_files(SR_LC8_files)
bt_brick <- brick_files(BT_LC8_files)

#Load CORINE Land Cover CLC2018 and project area of interest (AoI)
#Please use https://land.copernicus.eu/pan-european/corine-land-cover/clc2018 to obtain the data
CLC2018_V2018_20 <- raster("data/clc2018_clc2018_v2018_20_raster100m/CLC2018_CLC2018_V2018_20.tif")
CLC2018_V2018_20_AoI <- projectRaster(CLC2018_V2018_20, sr_brick, method= "ngb")
#You can check AoI values with unique(CLC2018_V2018_20_AoI) and distribution freq(CLC2018_V2018_20_AoI)

#Stack and brick all layers to ensure correct transformation, dimensions and resolution 
all_stack <- stack(sr_brick, bt_brick, CLC2018_V2018_20_AoI)
all_brick <- brick(all_stack)

# Set Irish projection keeping resolution and classification
#all_brick_ie <- projectRaster(all_brick, res = c(30,30), dim = c(1000,975), method="ngb", crs = "+init=epsg:2157 +proj=tmerc +lat_0=53.5 +lon_0=-8 +k=0.99982 +x_0=600000 +y_0=750000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")

#Prepare data only (without CORINE)
data_brick <- stack(sr_brick, bt_brick)
data_brick <- brick(data_brick)

#Visualize results
#mapview(all_brick)

#PREPARE FOR ML
#Set number of pixels from each side
r <- 5

#Process all data
process_row <-dim(data_brick)[1]
#or limit row numbers
process_row <-20

i=0 #number of rows
j=0 #number of columns
number_bogs<-0 #specific land cover type  
number_nonbogs<-0 #all others

train_data <- NULL #train data
train_categories <- c() #train categories

for (x in (1+r):(process_row-r)){
  cat("Row X:",x,"\n")
  for (y in (1+r):(dim(data_brick)[2]-r)){
    cat("Column Y:",y,"\n")
    
    #crop AoI
    tmp <- crop(data_brick,  extent(data_brick, x-r, x+r, y-r, y+r))
    tmp_array <- as.array(tmp)
    #new vector for Keras ML
    new_vector <- array_reshape(tmp_array, c(1, ((1+r+r)*(1+r+r)*9)))
    i = i+1
    if (all_brick[x,y][10]==412||all_brick[x,y][10]==411){
      number_bogs<-number_bogs+1
      train_data <- array( c( train_data , new_vector) , dim = c(number_bogs+number_nonbogs, ((1+r+r)*(1+r+r)*9) ))
      train_categories <- c(train_categories,1) 
      }
    else { 
      # uncomment next line if needed a balanced representation
      # if (number_bogs > number_nonbogs){
      number_nonbogs<-number_nonbogs+1
      train_data <- array( c( train_data , new_vector) , dim = c(number_bogs+number_nonbogs, ((1+r+r)*(1+r+r)*9) ))
      train_categories <- c(train_categories,0)
      #uncomment next line if needed a balanced representation
      #}
    }
  }
}

#You can save prepared data
#saveRDS(train_data, "data/traindat")
#saveRDS(train_categories, "data/traincat")

#You can load previously prepared data
#train_categories <- readRDS("data/traindat")
#train_data <- readRDS("data/traincat")

#PREPARE TEST DATA FOR ML
#r <- 5 #keep it the same

process_row <-dim(data_brick)[1]

#change row number to the AoI
process_row <- 210 
i=0
j=0
k=0
test_data <- NULL
test_categories <- c()
for (x in (200+r):(process_row-r)){
  cat("X:",x,"\n")
  for (y in (1+r):(dim(data_brick)[2]-r)){
    cat("Y:",y,"\n")
    #crop AoI
    tmp <- crop(data_brick,  extent(data_brick, x-r, x+r, y-r, y+r))
    tmp_array <- as.array(tmp)
    #new vector
    new_vector <- array_reshape(tmp_array, c(1, ((1+r+r)*(1+r+r)*9)))
    i = i+1
    if (all_brick[x,y][10]==412||all_brick[x,y][10]==411){
      test_data <- array( c( test_data, new_vector) , dim = c(i, ((1+r+r)*(1+r+r)*9) ))
      test_categories <- c(test_categories,1)
      k<k+1
    }
    else {
      #   if (k > (i-k)){
      test_data <- array( c( test_data, new_vector) , dim = c(i, ((1+r+r)*(1+r+r)*9) ))
      test_categories <- c(test_categories,0)
      #  }
    }
  }
}

#ML

#R+Keras Installation Notes:
#install.packages("devtools")
#devtools::install_github("rstudio/keras")
#devtools::install_github("rstudio/tensorflow")
#install_keras() 

r<-5 #keep it the same across the code

#Load predefined dataset
#x_train <- readRDS("traindat50r1")
#y_train <- readRDS("traincat50r1")

x_train <- train_data
y_train <- train_categories

# rescale if needed
# x_train <- abs(x_train+2000)/10000

#
y_train <- to_categorical(y_train, 2)

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 32, activation = 'relu', input_shape = c(((1+r+r)*(1+r+r)*9))) %>% 
  # layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 16, activation = 'relu') %>%
  # layer_dropout(rate = 0.1) %>%
  layer_dense(units = 2, activation = 'softmax')

history <- model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

summary(model)
history <- model %>% fit(
  x_train, y_train, 
  epochs = 25, batch_size = 50, 
  validation_split = 0.2
)

#VALIDATION TEST DATA

#You can preload previously saved data:
#x_test <- readRDS("traindat50r1")
#y_test <- readRDS("traincat50r1")

x_test <- test_data
y_test <- test_categories
y_test <- to_categorical(y_test, 2)

#Get results
model %>% evaluate(x_test, y_test)

