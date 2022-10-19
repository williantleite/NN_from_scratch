library(imager)
filenames_cat <- list.files("C:/Users/wtrindad/source/repos/NN_from_scratch/PetImages/Cat/", pattern="*.jpg", full.names=T)
filenames_dog <- list.files("C:/Users/wtrindad/source/repos/NN_from_scratch/PetImages/Dog/", pattern="*.jpg", full.names=T)
load <- function(im){
  skip_to_next<-FALSE
  tryCatch(load.image(im), error = function(e){skip_to_next<<-TRUE},
           if(skip_to_next){next})
}
clean_set<-function(loaded_images){
  for(i in 1:length(loaded_images)){
    error_check<-FALSE
    tryCatch(if(dim(loaded_images[[i]])[4]!=3){
      loaded_images <- loaded_images[-i]
    }, error=function(e){error_check<<-TRUE},
    if(error_check){loaded_images=loaded_images[-i]})
    tryCatch(if(dim(loaded_images[[i]])[4]==3){ 
      next},error=function(e){error_check<<-TRUE},
    if(error_check){loaded_images=loaded_images[-i]})
    tryCatch(if(loaded_images[[i]]==1){
      loaded_images <- loaded_images[-1]
    }, error=function(e){error_check<<-TRUE},
    if(error_check){next})
  }
  return(loaded_images)
}
resize_set<-function(loaded_images,width,height){
  for(i in 1:length(loaded_images)){
    loaded_images[[i]]<-resize(loaded_images[[i]],width,height,interpolation_type=1)
  }
  loaded_images<-do.call("cbind",loaded_images)
  return(loaded_images)
}

create_dataset<-function(filenames_cat,filenames_dog, labels=c(1,0),width,height){
  loaded_cats<-sapply(filenames_cat,FUN=load,USE.NAMES=TRUE)
  names(loaded_cats)<-gsub("C:/Users/wtrindad/source/repos/NN_from_scratch/PetImages/","",names(loaded_cats))
  plot(loaded_cats$`Cat/1.jpg`)
  loaded_cats<-clean_set(loaded_cats)
  loaded_cats<-resize_set(loaded_cats,width,height)
  cat_labels<-rep(label[1],dim(loaded_cats)[2])
  plot(as.cimg(loaded_cats[,2],x=32,y=32,z=1,cc=3))
  loaded_dogs<-sapply(filenames_dog,FUN=load,USE.NAMES=TRUE)
  names(loaded_dogs)<-gsub("C:/Users/wtrindad/source/repos/NN_from_scratch/PetImages/","",names(loaded_dogs))
  plot(loaded_dogs$`Dog/1.jpg`)
  loaded_dogs<-clean_set(loaded_dogs)
  loaded_dogs<-resize_set(loaded_dogs,width=32,height=32)
  dog_labels<-rep(label[2],dim(loaded_dogs)[2])
  plot(as.cimg(loaded_dogs[,2],x=32,y=32,z=1,cc=3))
  dataset<-cbind(loaded_cats,loaded_dogs)
  labels<-append(cat_labels,dog_labels)
  return(dataset,labels)
}
img_data<-create_dataset(filenames_cat,filenames_dog,labels=c(1,0),width=32,height=32)

test_mat<-do.call("cbind",loaded_dogs)
