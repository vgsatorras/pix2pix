--[[
---DENSNET
function BN_ReLU_Conv(inputs, n_filters_in, n_filters_out, filter_size, dropout_p)
    if filter_size == nil then
        filter_size = 3
    end
    if dropout_p == nil then
        dropout_p = 0.2
    end

    x = inputs - nn.SpatialBatchNormalization(n_filters_in) - nn.ReLU(true)
    x = x - nn.SpatialConvolution(n_filters_in, n_filters_out, filter_size, filter_size, 1, 1, 1, 1)

    if dropout_p ~= 0 then
        x = x - nn.Dropout(dropout_p)
    end
    return x
end 


function TransitionDown(inputs, n_filters_in, n_filters_out, dropout_p)
    ---Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2 

    if dropout_p == nil then
        dropout_p = 0.2
    end
    x = BN_ReLU_Conv(inputs, n_filters_in, n_filters_out, 1, dropout_p)
    x = x - nn.SpatialMaxPooling(2,2,2,2)
    return x

    --Note : network accuracy is quite similar with average pooling or without BN - ReLU.
    --We can also reduce the number of parameters reducing n_filters in the 1x1 convolution
end



function TransitionUp(skip_connection, block_to_upsample, n_filters_input, n_filters_keep)
    --Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection
    --# Upsample
    --x = block_to_upsample - nn.JoinTable(2)
    x = block_to_upsample - nn.SpatialConvolution(n_filters_input, n_filters_keep, 4, 4, 2, 2, 1, 1)
    x = {x, skip_connection} - nn.JoinTable(2)

    return x
    -- Note : we also tried Subpixel Deconvolution without seeing any improvements.
    -- We can reduce the number of parameters reducing n_filters_keep in the Deconvolution
end

function defineG_densenet(input_nc, output_nc, n_filters_first_conv, n_classes, n_pool, growth_rate, n_layers_per_block, dropout_p)
    --
    --- input_nc            1
    --- output_nc           2
    --- ngf                 48
    --- n_classes           2xx
    --- n_pool              5
    --- growth_rate         16
    --- n_layers_per_block  {4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4}
    --
    nngraph.setDebug(true)


    if dropout_p == nil then
        dropout_p = 0.2
    end

    if type(n_layers_per_block) == type({}) then
        assert (#n_layers_per_block == 2 * n_pool + 1)
    elseif type(n_layers_per_block) == type(1) then
        num = n_layers_per_block
        n_layers_per_block = {}
        for i= 1,(2 * n_pool + 1) do
            table.insert(n_layers_per_block, num)
        end
    else
        assert(false)
    end

    ------------------
    -- DOWNSAMPLING --
    ------------------

    input = - nn.SpatialConvolution(input_nc, n_filters_first_conv, 3, 3, 1, 1, 1, 1)
    stack = input
    n_filters = n_filters_first_conv

    skip_connection_list = {}
    skip_connection_list_n_features = {}

    for i = 1,n_pool do
        -- Dense Block
        for j = 1,n_layers_per_block[i] do
            -- Compute new feature maps
            x = BN_ReLU_Conv(stack, n_filters, growth_rate, 3, dropout_p)
            -- And stack it : the Tiramisu is growing
            stack = {stack, x} - nn.JoinTable(2)
            n_filters = n_filters + growth_rate
        end
        -- At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list[i] = {stack} - nn.JoinTable(2) 
        table.insert(skip_connection_list_n_features, n_filters)

        -- Transition Down
        stack = TransitionDown(stack, n_filters, n_filters, dropout_p)
    end


    print ("Len skipconnection " .. #skip_connection_list)
    skip_connection_list = util.list_reverse(skip_connection_list)
    print ("Len skipconnection " .. #skip_connection_list)
    skip_connection_list_n_features = util.list_reverse(skip_connection_list_n_features)

    ----------------
    -- BOTTLENECK --
    ----------------

    --We store now the output of the next dense block in a list. We will only upsample these new feature maps
    block_to_upsample = nil
    features_before_dense_block = n_filters
    --Dense Block
    for j =1,n_layers_per_block[n_pool+1] do
        x = BN_ReLU_Conv(stack, n_filters, growth_rate, 3, dropout_p)

        if block_to_upsample == nil then
            block_to_upsample = x
        else
            block_to_upsample = {block_to_upsample, x} - nn.JoinTable(2)
        end
        --table.insert(block_to_upsample, x)

        if j < n_layers_per_block[n_pool+1] then 
            stack = {stack, x} - nn.JoinTable(2)
        end
        n_filters = n_filters + growth_rate
    end
    features_after_dense_block = n_filters


    -- UNTIL THIS POINTS, IT COMPILES

    --------------------
    -- CLASSIFICATION --
    --------------------
    
    --x = stack - nn.ReLU(true) - nn.SpatialConvolution(n_filters, 512, 3, 3, 2, 2, 1, 1) - nn.Dropout(0.2) - nn.ReLU(true)
    --x = x - nn.View(4*4*512)
    --x = x - nn.Linear(4*4*512, 768) - nn.Dropout(0.2) - nn.ReLU(true)
    --o_class = x - nn.Linear(768, 512) - nn.Dropout(0.2) - nn.ReLU(true) - nn.Linear(512, n_classes) - nn.LogSoftMax()
    
    
    ----------------
    -- UPSAMPLING --
    ----------------
    current_features = features_after_dense_block - features_before_dense_block

    --aux = skip_connection_list - nn.JoinTable(2)
    --netG = nn.gModule({input},{block_to_upsample, aux})
    --print ("Compiles 1")


    for i = 1,n_pool do
        -- Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]


        stack = TransitionUp(skip_connection_list[i], block_to_upsample, current_features, n_filters_keep)
        current_features = n_filters_keep + skip_connection_list_n_features[i]


        aux = skip_connection_list - nn.JoinTable(2)
        print (stack)
        print (aux)
        netG = nn.gModule({input},{aux, stack})
        print ("Compiles ".. i+1)

        --Dense Block
        block_to_upsample = nil
        print ("Dense block " .. n_pool + i + 1 .. " layers: " .. n_layers_per_block[n_pool + i + 1])
        for j = 1,n_layers_per_block[n_pool + i + 1] do
            x = BN_ReLU_Conv(stack, current_features, growth_rate, 3, dropout_p)
            n_filters = n_filters + growth_rate
            current_features = current_features + growth_rate

            if i < n_pool then
                if block_to_upsample == nil then
                    block_to_upsample = x
                else
                    block_to_upsample = {block_to_upsample, x} - nn.JoinTable(2)
                end
                --table.insert(block_to_upsample, x)
            end

            stack = {stack, x} - nn.JoinTable(2)
        end





    end

    o1 = stack - nn.ReLU(true) - nn.SpatialFullConvolution(current_features, output_nc, 1, 1, 1, 1, 0, 0) - nn.Tanh()
    
    nngraph.annotateNodes()
    netG = nn.gModule({input},{o1})
    print ("COMPILES!")
end
---DENSENET
--]]
