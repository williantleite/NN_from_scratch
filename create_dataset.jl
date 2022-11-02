using Images, FileIO, InvertedIndices, Suppressor

function process_image(path_vec::Vector{String}, h::Int64, w::Int64, label::Int64)
  result = zeros((h*w*3), length(path_vec))
  class = Int[]
  @suppress begin
    for i in enumerate(path_vec) 
      try
        img = load(i[2])
      catch 
        continue
      end
      img = (img === nothing) ? continue : img
      img = imresize(img,(h,w))
      img = size(img) == (h, w) ? img : continue
      img = vec(img)
      try
        img = [temp(img[i]) for i = 1:length(img), temp in (red, green, blue)]
      catch
        continue
      end
      img = reshape(img, ((h*w*3),1))
      result[:,i[1]] = img
      push!(class, label)
    end
  end
  return result, class
end

function create_dataset(filenames_cat::Vector{String}, filenames_dog::Vector{String}, height::Int64, width::Int64, labels)
  cat_i, cat_l = process_image(filenames_cat, height, width, labels[1])
  dog_i, dog_l = process_image(filenames_dog, height, width, labels[2])
  imgs = hcat(cat_i, dog_i)
  class = vcat(cat_l, dog_l)
  i=1
  while i <= size(imgs)[2]
    imgs = sum(imgs[:,i]) == 0 ? imgs[:, Not(i)] : imgs
    i +=1
  end
  return imgs, class
end