clear all

T = [0.005 0.007 0.012 0.015 0.023 0.030 0.026 0.024 0.019 0.010 0.004 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000;...
     0.000 0.000 0.000 0.000 0.000 0.001 0.002 0.003 0.005 0.007 0.012 0.013 0.015 0.016 0.017 0.020 0.013 0.011 0.009 0.005 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.002 0.002 0.003;...
     0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.001 0.003 0.010 0.012 0.013 0.022 0.020 0.020 0.018 0.017 0.016 0.016 0.014 0.014 0.013];
ST=sum(T,2);

load_imgDataPath = '.\Result\ICVL'
savePath = '.\Result\ICVL_HSI\result';
size_input = 48;


hyperspectralSlices =1:31;

stride = size_input/2;

addpath(load_imgDataPath)
addpath(savePath)
imgDataDir  = dir(load_imgDataPath);            
Result_psnr_ssim=zeros(length(imgDataDir)-2,3,'single');
for i = 1:length(imgDataDir)
    if(isequal(imgDataDir(i).name,'.')||... 
       isequal(imgDataDir(i).name,'..'))
      continue;
    end
    load([load_imgDataPath,'/',imgDataDir(i).name]);
    imgName = imgDataDir(i).name;
    imgName = imgName(1:end-4);
    output = output.*(output>0);

        
    %reconstruct image
    count=1;
    result_Image = zeros(size(label));
    result_Weight = zeros(size(label));
    [hei,wid,ch]=size(label);
    kh=1;
    kw=1;
    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1 
            temp_out = output(:,:,:,count);
            result_Image(x: x+size_input-1, y : y+size_input-1,:)=result_Image(x: x+size_input-1, y : y+size_input-1,:)...
                +temp_out;
            result_Weight(x: x+size_input-1, y : y+size_input-1,:)=result_Weight(x: x+size_input-1, y : y+size_input-1,:)...
                +1;
            count=count+1;
            kw=y;
        end
        kh=x;
    end
    label = label(1:kh+size_input-1,1:kw+size_input-1,:);
    result_Image = result_Image(1:kh+size_input-1,1:kw+size_input-1,:);
    result_Weight = result_Weight(1:kh+size_input-1,1:kw+size_input-1,:);

    for ch=1:31
        label(:,:,ch) = circshift(label(:,:,ch),[(ch-1),0]);
        result_Image(:,:,ch) = circshift(result_Image(:,:,ch),[(ch-1),0]);
        result_Weight(:,:,ch) = circshift(result_Weight(:,:,ch),[(ch-1),0]);
    end
    label(1:30, :, :)=[];
    result_Image(1:30, :, :)=[];
    result_Weight(1:30, :, :)=[];
    x_recon = result_Image./(result_Weight+eps);
    result = x_recon.*(x_recon>0);
   
    
	%Reconstruct the RGB image
	rgb_recon=zeros(size(result,1),size(result,2),3);
    rgb_gt=zeros(size(result,1),size(result,2),3);
    gt=label;
    x_recon=result;
    for ch=1:31
        x_recon(:,:,ch)=x_recon(:,:,ch)./(max(max(gt(:,:,ch))));
        gt(:,:,ch)=gt(:,:,ch)./(max(max(gt(:,:,ch))));   
    end
    for ch=1:31
        rgb_recon(:,:,1)=rgb_recon(:,:,1)+T(1,ch)*x_recon(:,:,ch);
        rgb_recon(:,:,2)=rgb_recon(:,:,2)+T(2,ch)*x_recon(:,:,ch);
        rgb_recon(:,:,3)=rgb_recon(:,:,3)+T(3,ch)*x_recon(:,:,ch);
        
        rgb_gt(:,:,1)=rgb_gt(:,:,1)+T(1,ch)*gt(:,:,ch);
        rgb_gt(:,:,2)=rgb_gt(:,:,2)+T(2,ch)*gt(:,:,ch);
        rgb_gt(:,:,3)=rgb_gt(:,:,3)+T(3,ch)*gt(:,:,ch);
      
    end
    for ch=1:3
        tt=ST(ch);
        rgb_recon(:,:,ch)=rgb_recon(:,:,ch)./tt;
        rgb_gt(:,:,ch)=rgb_gt(:,:,ch)./tt;
    end
    temp=rgb_recon(:,:,1);
    rgb_recon(:,:,1)=rgb_recon(:,:,3);
    rgb_recon(:,:,3)=temp;
    temp=rgb_gt(:,:,1);
    rgb_gt(:,:,1)=rgb_gt(:,:,3);
    rgb_gt(:,:,3)=temp;
    
    filleName=fullfile(savePath,'RGB',imageName);
    if ~exist(filleName,'file')
        mkdir(filleName);
    end
    imwrite(rgb_recon,[filleName,'\rgb_ours.png']);
    imwrite(rgb_gt,[filleName,'\rgb_gt.png']);
	
 
    %Calculate PSNR,SSIM,SAM
    h = size(result,1);
    w = size(result,2);
    ground_truth = label;
    img_result=result;
    psnr=zeros(1,31);
    for k = 1:31
        img_max = max(max(ground_truth(:,:,k)));
        err = mean(mean((ground_truth(:,:,k)-img_result(:,:,k)).^2));
        psnr(k) = 10*log10(img_max^2/err);
    end
    PSNR = mean(psnr);
    
    ssim=zeros(1,31);
    k1 = 0.01;
    k2 = 0.03;
    for k = 1:31
        a = 2*mean(mean(img_result(:,:,k))) * mean(mean(ground_truth(:,:,k))) + k1^2;
        x = cov(reshape(img_result(:,:,k), h*w, 1), reshape(ground_truth(:,:,k), h*w, 1));
        b = 2*x(1,2) + k2^2;
        c = mean(mean(img_result(:,:,k)))^2 + mean(mean(ground_truth(:,:,k)))^2 + k1^2;
        d = x(1,1) + x(2,2) + k2^2;
        ssim(k) = a*b/c/d;
    end
    SSIM = mean(ssim);
    
    tmp = (sum(ground_truth.*img_result, 3) + eps) ...
    ./ (sqrt(sum(ground_truth.^2, 3)) + eps) ./ (sqrt(sum(img_result.^2, 3)) + eps);
    SAM = mean2(real(acos(tmp)));
    
    
    disp([num2str(i-2), '-', num2str(PSNR), '-', num2str(SSIM), '-', num2str(SAM)]);
    Result_psnr_ssim(i-2,:)=[PSNR,SSIM,SAM];
	
	savedir=fullfile(savePath,imgName);
    if ~exist(savedir,'file')
        mkdir(savedir);
    end
	save(fullfile(savedir,'result'),'result','label','PSNR','SSIM','SAM');
end
























