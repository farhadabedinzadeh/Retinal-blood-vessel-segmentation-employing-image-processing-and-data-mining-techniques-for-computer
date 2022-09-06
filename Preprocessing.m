function Feature = Preprocessing(im,counter)
    ImCrop = im(30:564, 15:550,1:3);
     [lp]= createmask(ImCrop);
    % % % =========================== Trnasformation Phase 
    %%=============================1: Lab transformation 
     colorTransform = makecform('srgb2lab');
    lab = applycform(ImCrop,colorTransform);
    %%%=============================2,3: YCbCr trnasformation and Guassian Trnasformation
    transformation = [0.257, 0.504, 0.098; -0.148, -0.291 , 0.439 ; 0.439 -0.368 -0.071];
    Gaussian = [0.06, 0.63, 0.27;0.3 , 0.04 , -0.35; 0.34 , -0.6 0.17];
    YCbCr = ImCrop;
    GImage = ImCrop;
    [Ix Iy Iz] = size(ImCrop);
    Temp = zeros(3,1);
    for i = 1:Ix
        for j = 1:Iy
            Temp(1) = ImCrop(i,j,1);
            Temp(2) = ImCrop(i,j,2);
            Temp(3) = ImCrop(i,j,3);
            YCbCr(i,j,:) = transformation * Temp + [16;128;128];
            GImage(i,j,:) = Gaussian * Temp;
        end 
    end
    %%%% ================================= Final Trnasformation Image
    GRB=ImCrop(:,:,2);
    F = YCbCr(:,:,1);
    D = lab(:,:,1);
    G = GImage(:,:,1);
    %%%=================== CLAHE (Contrast-limited Adaptive Histogram Equalization) algorithm for contrast Enhancement
    Green =  adapthisteq(GRB,'clipLimit',0.01,'Distribution','uniform');
    Y =  adapthisteq(F,'clipLimit',0.01,'Distribution','uniform');
    L =  adapthisteq(D,'clipLimit',0.01,'Distribution','uniform');
    G1 =  adapthisteq(G,'clipLimit',0.01,'Distribution','uniform');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gabor Filtering on the 4 Channels (Green,Y,L,G1 )
    Input = cell(4,1);
    Input{1} = G;
    Input{2} = G1;
    Input{3} = Y;
    Input{4} = Green;
    theta   = 0;    %%%% theta -> angle in rad
    lambda  = [9 10 11];   %%%%% lambda -> Wave Length
    psi     = [-pi pi]; %%% psi ->phase shift
    gamma   = 0.5; %%%%% gamma ->aspect ration
    bw      = 1;   %%% bw -> bandwidth
    N       = 24;
    % % % 
    Feature = zeros(size(G1,1),size(G1,2),13);
    for i =1:4
        for j = 1:3
        KHSH  = Input{i};
       GG_example = gabor_example(KHSH,lambda(j),theta,psi,gamma,bw,N);
       maxIntensify = max(max(GG_example));
       perVal = maxIntensify * .1;
       GG_example(GG_example<=perVal)=0;
       GG_example(GG_example>perVal)=1; 
     GG_final = (lp .*(GG_example));
       Feature(:,:,(i-1)*3 + j) = GG_final;   
        end
    end
    Green = double(Green)/255;
    Feature(:,:,13) = Green;
end