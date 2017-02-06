require 'nngraph'




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
    --[[
    input_nc            1
    output_nc           2
    ngf                 48
    n_classes           2xx
    n_pool              5
    growth_rate         16
    n_layers_per_block  {4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4}
    ]]--
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
    --[[
    x = stack - nn.ReLU(true) - nn.SpatialConvolution(n_filters, 512, 3, 3, 2, 2, 1, 1) - nn.Dropout(0.2) - nn.ReLU(true)
    x = x - nn.View(4*4*512)
    x = x - nn.Linear(4*4*512, 768) - nn.Dropout(0.2) - nn.ReLU(true)
    o_class = x - nn.Linear(768, 512) - nn.Dropout(0.2) - nn.ReLU(true) - nn.Linear(512, n_classes) - nn.LogSoftMax()
    --]]
    
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
        netG = nn.gModule({input},{aux, stack2})
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

function defineG_unet(input_nc, output_nc, ngf, n_class)
    
    -- input is (nc) x 256 x 256
    input = - nn.SpatialConvolution(input_nc, ngf, 3, 3, 2, 2, 1, 1)
    e1 = input - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x 128 x 128
    x = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    e2 = x - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf*2, ngf * 2, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    x = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    e3 = x - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    x = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    e4 = x - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    x = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    e5 = x - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 4, 4, 1, 1, 0, 0) - nn.SpatialBatchNormalization(ngf * 16) - nn.Dropout(0.2)
    -- input is (ngf * 8) x 2 x 2
    --e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 16)
    -- input is (ngf * 8) x 1 x 1
    --x = e7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.2)
    x = e7 - nn.SpatialUpSamplingNearest(2)
    d1_ = x - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.2)
    
    -- input is (ngf * 8) x 2 x 2
    --d1 = {d1_,e7} - nn.JoinTable(2)
    --d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.2)
    
    -- input is (ngf * 8) x 4 x 4
    d1 = {d1_, e6} - nn.JoinTable(2)
    d3_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.2)
    -- input is (ngf * 8) x 8 x 8
    d3 = {d3_,e5} - nn.JoinTable(2)
    x = d3 - nn.ReLU(true) - nn.SpatialConvolution(ngf * 8 * 2, ngf * 8, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    d4_ = x - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    d4 = {d4_,e4} - nn.JoinTable(2)
    x = d4 - nn.ReLU(true) - nn.SpatialConvolution(ngf * 8 * 2, ngf * 4, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    d5_ = x - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    d5 = {d5_,e3} - nn.JoinTable(2)
    x = d5 - nn.ReLU(true) - nn.SpatialConvolution(ngf * 4 * 2, ngf * 2, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    d6_ = x - nn.ReLU(true) - nn.SpatialFullConvolution(ngf *2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    d6 = {d6_,e2} - nn.JoinTable(2)
    x = d6 - nn.ReLU(true) - nn.SpatialConvolution(ngf * 2 * 2, ngf, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf)
    --d7_ = x - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    d7 = x - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x128 x 128
    --d7 = {d7_,e1} - nn.JoinTable(2)
    --d8 = d7_ - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)

    -- input is (nc) x 256 x 256
    
    o1 = d7 - nn.Tanh()


    x = e7 - nn.ReLU(true) - nn.View(1024) 
    x = x - nn.Linear(1024, 768) - nn.Dropout(0.2) - nn.ReLU(true)
    o_class = x - nn.Linear(768, 512) - nn.Dropout(0.2) - nn.ReLU(true) - nn.Linear(512, n_class) - nn.LogSoftMax()
    
    netG = nn.gModule({input},{o1, o_class})
    
    --graph.dot(netG.fg,'netG')
    
    return netG
end



function defineG_unet_old(input_nc, output_nc, ngf, n_class)
    
    -- input is (nc) x 256 x 256
    input = - nn.SpatialConvolution(input_nc, ngf, 5, 5, 2, 2, 1, 1)
    e1 = input - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x 128 x 128
    x = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    e2 = x - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf*2, ngf * 2, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    x = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    e3 = x - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    x = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    e4 = x - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    x = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    e5 = x - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    x = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    e6 = x - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    x = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    e7 = x - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 3, 3, 1, 1, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 16, 3, 3, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 16)
    -- input is (ngf * 8) x 1 x 1
    
    d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 16, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.4)
    -- input is (ngf * 8) x 2 x 2
    d1 = {d1_,e7} - nn.JoinTable(2)
    d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.4)
    -- input is (ngf * 8) x 4 x 4
    d2 = {d2_,e6} - nn.JoinTable(2)
    d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.4)
    -- input is (ngf * 8) x 8 x 8
    d3 = {d3_,e5} - nn.JoinTable(2)
    d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    d4 = {d4_,e4} - nn.JoinTable(2)
    d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    d5 = {d5_,e3} - nn.JoinTable(2)
    d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    d6 = {d6_,e2} - nn.JoinTable(2)
    d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    --d7 = {d7_,e1} - nn.JoinTable(2)
    d8 = d7_ - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    
    o1 = d8 - nn.Tanh()


    x = e8 - nn.Dropout(0.2) - nn.ReLU(true) - nn.View(1024) 
    x = x - nn.Linear(1024, 768) - nn.Dropout(0.2) - nn.ReLU(true)
    o_class = x - nn.Linear(768, 512) - nn.Dropout(0.2) - nn.ReLU(true) - nn.Linear(512, n_class) - nn.LogSoftMax()
    
    netG = nn.gModule({input},{o1, o_class})
    
    --graph.dot(netG.fg,'netG')
    
    return netG
end



function defineG_unet_(input_nc, output_nc, ngf)
    
    -- input is (nc) x 256 x 256
    e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1
    
    d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    d1 = {d1_,e7} - nn.JoinTable(2)
    d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    d2 = {d2_,e6} - nn.JoinTable(2)
    d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    d3 = {d3_,e5} - nn.JoinTable(2)
    d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    d4 = {d4_,e4} - nn.JoinTable(2)
    d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    d5 = {d5_,e3} - nn.JoinTable(2)
    d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    d6 = {d6_,e2} - nn.JoinTable(2)
    d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    d7 = {d7_,e1} - nn.JoinTable(2)
    d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    
    o1 = d8 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})
    
    --graph.dot(netG.fg,'netG')
    
    return netG
end


function defineC(n_class)
    local netC = nn.Sequential()
    netC:add(nn.ReLU(true))

    netC:add(nn.View(1024))
    netC:add(nn.Linear(1024, 768))
    netC:add(nn.Dropout(0.2))
    netC:add(nn.ReLU(true))

    netC:add(nn.Linear(768, 512))
    netC:add(nn.Dropout(0.2))
    netC:add(nn.ReLU(true))

    netC:add(n.Linear(512, n_class))
    netC:add(nn.LogSoftMax())

    return netC

end


function defineD_sn(netD, n)
    local netD_sn = nn.Sequential()
    for i = 1,n do
        netD_sn:add(nn.SpatialAveragePooling(2,2,2,2))
    end
    netD_sn:add(netD)
    return netD_sn
end 

function defineD_s4(netD)
    local netD_s4 = nn.Sequential()
    netD_s4:add(nn.SpatialAveragePooling(2,2,2,2))
    netD_s4:add(nn.SpatialAveragePooling(2,2,2,2))
    netD_s4:add(netD)
    return netD_s4
end

function define_upsampler_net()
    local upsampler_net = nn.Sequential()
    upsampler_net:add(nn.SpatialUpSamplingNearest(2))
    return upsampler_net
end

function define_downsampler_net()
    local downsampler_net = nn.Sequential()
    downsampler_net:add(nn.SpatialAveragePooling(2,2,2,2))
    return downsampler_net
end

function defineD_basic(input_nc, output_nc, ndf)
    
    n_layers = 3
    return defineD_n_layers(input_nc, output_nc, ndf, n_layers)
end

-- rf=1
function defineD_pixelGAN(input_nc, output_nc, ndf)
    
    local netD = nn.Sequential()
    
    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 1, 1, 1, 1, 0, 0))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 1, 1, 1, 1, 0, 0))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf * 2, 1, 1, 1, 1, 1, 0, 0))
    -- state size: 1 x 256 x 256
    
    netD:add(nn.Sigmoid())
    -- state size: 1 x 30 x 30
        
    return netD
end



-- if n=0, then use pixelGAN (rf=1)
-- else rf is 16 if n=1
--            34 if n=2
--            70 if n=3
--            142 if n=4
--            286 if n=5
--            574 if n=6
function defineD_n_layers(input_nc, output_nc, ndf, n_layers, classes_disc)
    
    if n_layers==0 then
        return defineD_pixelGAN(input_nc, output_nc, ndf)
    else
    
        local netD = nn.Sequential()
        
        -- input is (nc) x 256 x 256
        netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
        netD:add(nn.LeakyReLU(0.2, true))
        
        nf_mult = 1
        for n = 1, n_layers-1 do 
            nf_mult_prev = nf_mult
            nf_mult = math.min(2^n,8)
            netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
            netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        end
        
        -- state size: (ndf*M) x N x N
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n_layers,8)
        netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, math.floor(ndf * nf_mult * 1.5), 4, 4, 2, 2, 1, 1))
        netD:add(nn.SpatialBatchNormalization(math.floor(ndf * nf_mult * 1.5))):add(nn.LeakyReLU(0.2, true))
        -- state size: (ndf*M*2) x (N-1) x (N-1)
        netD:add(nn.SpatialConvolution(math.floor(ndf * nf_mult * 1.5), classes_disc, 4, 4, 1, 1, 0, 0))
        -- state size: 1 x (N-2) x (N-2)
        
        netD:add(nn.Sigmoid())
        -- state size: 1 x (N-2) x (N-2)
        
        return netD
    end
end