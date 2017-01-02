--
-- code derived from https://github.com/soumith/dcgan.torch
--

local util = {}

require 'torch'

function util.normalize(img)
  -- rescale image to 0 .. 1
  local min = img:min()
  local max = img:max()
  
  img = torch.FloatTensor(img:size()):copy(img)
  img:add(-min):mul(1/(max-min))
  return img
end

function util.normalizeBatch(batch)
	for i = 1, batch:size(1) do
		batch[i] = util.normalize(batch[i]:squeeze())
	end
	return batch
end

function util.basename_batch(batch)
	for i = 1, #batch do
		batch[i] = paths.basename(batch[i])
	end
	return batch
end



-- default preprocessing
--
-- Preprocesses an image before passing it to a net
-- Converts from RGB to BGR and rescales from [0,1] to [-1,1]
function util.preprocess(img)
    -- RGB to BGR
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm)
    
    -- [0,1] to [-1,1]
    img = img:mul(2):add(-1)
    
    -- check that input is in expected range
    assert(img:max()<=1,"badly scaled inputs")
    assert(img:min()>=-1,"badly scaled inputs")
    
    return img
end

-- Undo the above preprocessing.
function util.deprocess(img)
    
    -- BGR to RGB
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm)
    
    -- [-1,1] to [0,1]
    
    img = img:add(1):div(2)
    
    return img
end

function util.preprocess_batch(batch)
	for i = 1, batch:size(1) do
		batch[i] = util.preprocess(batch[i]:squeeze())
	end
	return batch
end

function util.deprocess_batch(batch)
	for i = 1, batch:size(1) do
		batch[i] = util.deprocess(batch[i]:squeeze())
	end
	return batch
end



-- preprocessing specific to colorization

function util.deprocessLAB(L, AB)
    local L2 = torch.Tensor(L:size()):copy(L)
    if L2:dim() == 3 then
      L2 = L2[{1, {}, {} }]
    end
    local AB2 = torch.Tensor(AB:size()):copy(AB)
    AB2 = torch.clamp(AB2, -1.0, 1.0)
--    local AB2 = AB
    L2 = L2:add(1):mul(50.0)
    AB2 = AB2:mul(110.0)
    
    L2 = L2:reshape(1, L2:size(1), L2:size(2))
    
    im_lab = torch.cat(L2, AB2, 1)
    im_rgb = torch.clamp(image.lab2rgb(im_lab):mul(255.0), 0.0, 255.0)/255.0
    
    return im_rgb
end

function util.deprocessL(L)
    local L2 = torch.Tensor(L:size()):copy(L)
    L2 = L2:add(1):mul(255.0/2.0)
    
    if L2:dim()==2 then
      L2 = L2:reshape(1,L2:size(1),L2:size(2))
    end
    L2 = L2:repeatTensor(L2,3,1,1)/255.0
    
    return L2
end

function util.deprocessL_batch(batch)
  local batch_new = {}
  for i = 1, batch:size(1) do
    batch_new[i] = util.deprocessL(batch[i]:squeeze())
  end
  return batch_new
end

function util.deprocessLAB_batch(batchL, batchAB)
  local batch = {}
  
  for i = 1, batchL:size(1) do
    batch[i] = util.deprocessLAB(batchL[i]:squeeze(), batchAB[i]:squeeze())
  end
  
  return batch
end


function util.scaleBatch(batch,s1,s2)
	local scaled_batch = torch.Tensor(batch:size(1),batch:size(2),s1,s2)
	for i = 1, batch:size(1) do
		scaled_batch[i] = image.scale(batch[i],s1,s2):squeeze()
	end
	return scaled_batch
end



function util.toTrivialBatch(input)
    return input:reshape(1,input:size(1),input:size(2),input:size(3))
end
function util.fromTrivialBatch(input)
    return input[1]
end



function util.scaleImage(input, loadSize)
    
    -- replicate bw images to 3 channels
    if input:size(1)==1 then
    	input = torch.repeatTensor(input,3,1,1)
    end
    
    input = image.scale(input, loadSize, loadSize)
    
    return input
end

function util.getAspectRatio(path)
	local input = image.load(path, 3, 'float')
	local ar = input:size(3)/input:size(2)
	return ar
end

function util.loadImage(path, loadSize, nc)

    local input = image.load(path, 3, 'float')
    
   input= util.preprocess(util.scaleImage(input, loadSize))
    
    if nc == 1 then
        input = input[{{1}, {}, {}}]
    end
    
    return input 
end



-- TO DO: loading code is rather hacky; clean it up and make sure it works on all types of nets / cpu/gpu configurations
function util.load(filename, opt)
	if opt.cudnn>0 then
		require 'cudnn'
	end
	local net = torch.load(filename)
	if opt.gpu > 0 then
		require 'cunn'
		net:cuda()
		
		-- calling cuda on cudnn saved nngraphs doesn't change all variables to cuda, so do it below
		if net.forwardnodes then
			for i=1,#net.forwardnodes do
				if net.forwardnodes[i].data.module then
					net.forwardnodes[i].data.module:cuda()
				end
			end
		end
		
	else
		net:float()
	end
	net:apply(function(m) if m.weight then 
	    m.gradWeight = m.weight:clone():zero(); 
	    m.gradBias = m.bias:clone():zero(); end end)
	return net
end

function util.cudnn(net)
	require 'cudnn'
	require 'util/cudnn_convert_custom'
	return cudnn_convert_custom(net, cudnn)
end



-- HANDLING LABELS

function util.split(str,sep)
  local array = {}
  local reg = string.format("([^%s]+)",sep)
  for mem in string.gmatch(str,reg) do
    table.insert(array, mem)
  end
  return array
end

function util.table_invert(t)
  local u = { }
  for k, v in pairs(t) do u[v] = k end
  return u
end

function util.get_labels()
  return {"bakery#shop", "forest_path", "office", "desert#vegetation", "forest_road", "train_railway", "shopfront", "nursery", "bedroom", "mansion", "beauty_salon", "schoolhouse", "coffee_shop", "rope_bridge", "shoe_shop", "racecourse", "courthouse", "hotel#outdoor", "fairway", "ruin", "cemetery", "abbey", "temple#south_asia", "marsh", "pulpit", "inn#outdoor", "skyscraper", "museum#indoor", "train_station#platform", "dam", "golf_course", "coast", "dock", "conference_room", "assembly_line", "basement", "pavilion", "bowling_alley", "veranda", "bar", "jail_cell", "harbor", "valley", "boat_deck", "phone_booth", "food_court", "trench", "rice_paddy", "sea_cliff", "volcano", "tower", "river", "dining_room", "auditorium", "ski_slope", "shed", "home_office", "creek", "rock_arch", "attic", "pantry", "waiting_room", "driveway", "swimming_pool#outdoor", "pasture", "excavation", "railroad_track", "dinette#home", "staircase", "closet", "aqueduct", "motel", "candy_store", "fountain", "playground", "construction_site", "crosswalk", "highway", "hot_spring", "stadium#football", "gas_station", "formal_garden", "living_room", "supermarket", "cockpit", "mountain", "gift_shop", "aquarium", "slum", "butte", "boxing_ring", "kitchen", "stage#indoor", "courtyard", "yard", "restaurant", "kitchenette", "plaza", "office_building", "pagoda", "field#wild", "rainforest", "castle", "mausoleum", "monastery#outdoor", "wheat_field", "sky", "amusement_park", "kindergarden_classroom", "vegetable_garden", "television_studio", "canyon", "restaurant_patio", "islet", "cafeteria", "lighthouse", "palace", "hospital", "galley", "subway_station#platform", "bayou", "restaurant_kitchen", "botanical_garden", "field#cultivated", "doorway#outdoor", "raft", "underwater#coral_reef", "stadium#baseball", "art_studio", "ice_skating_rink#outdoor", "chalet", "cottage_garden", "tree_farm", "market#outdoor", "cathedral#outdoor", "bookstore", "alley", "hotel_room", "desert#sand", "iceberg", "orchard", "airport_terminal", "hospital_room", "runway", "track#outdoor", "shower", "snowfield", "clothing_store", "residential_neighborhood", "garbage_dump", "kasbah", "bridge", "basilica", "crevasse", "campsite", "art_gallery", "amphitheater", "viaduct", "boardwalk", "banquet_hall", "bus_interior", "sandbar", "pond", "fire_escape", "herb_garden", "corn_field", "music_studio", "game_room", "parlor", "ski_resort", "corridor", "locker_room", "engine_room", "wind_farm", "temple#east_asia", "topiary_garden", "windmill", "fire_station", "ballroom", "ice_cream_parlor", "mountain_snowy", "medina", "conference_center", "water_tower", "watering_hole", "church#outdoor", "parking_lot", "arch", "lobby", "butchers_shop", "classroom", "dorm_room", "badlands", "building_facade", "picnic_area", "ocean", "swamp", "martial_arts_gym", "reception", "baseball_field", "bamboo_forest", "apartment_building#outdoor", "laundromat", "patio", "igloo"}
end

inverse_labels = {}
inverse_labels = util.table_invert(util.get_labels())

function util.get_inverse_labels()
  return inverse_labels
end

function util.inList(element, array)
  for i = 1,(#array)[1] do
    if array[i] == element then 
      return true
    end
  end
  return false
end

-- LISTS

function util.list_reverse(list)
  list_response = {}
  for i = #list,1,-1 do
    table.insert(list_response, list[i])
  end
  return list_response
end

--TIME
function util.sleep(s)
  local ntime = os.time() + s
  repeat until os.time() > ntime
end




-- Metrics

function util.topKacc(labels_pred, labels_real, K)
  -- labels_pred --> {{0.01, 0.9, ..., 0.2}, {...}}
  -- labels_real --> {3, 205, 50, ..., 27}
  correct = 0
  batch_size = (#labels_real)[1]
  for i = 1,batch_size do
    -- extract top 5 
    res, ind = labels_pred[i]:topk(K, true)
    if util.inList(labels_real[i], ind) then
      correct = correct + 1
    end
  end
  return correct/batch_size
end


return util

