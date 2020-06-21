function [y,lb]=bandp1(x,fpl,fph,fsl,fsh,Rp,Rs,fs,STYLE,Fig)
pl=2*fpl/fs;
ph=2*fph/fs;
sl=2*fsl/fs;
sh=2*fsh/fs;
wp=[pl,ph];
ws=[sl,sh];
if strcmp(STYLE,'butter')
    [n,wn] = buttord(wp,ws,Rp,Rs);  % Gives mimimum order of filter
    [b,a] = butter(n,wn);           % Butterworth filter design
elseif strcmp(STYLE,'cheby1')
    [n,wn]=cheb1ord(ws,wp,Rp,Rs);
    [b,a]=cheby1(n,Rp,wp);
elseif strcmp(STYLE,'cheby2')
    [n,wn]=cheb2ord(wp,ws,Rp,Rs);  % Gives mimimum order of filter
    [b,a]=cheby2(n,Rs,wn);        % Chebyshev Type II filter    
elseif strcmp(STYLE,'ellip')
    [n,wn]=ellipord(wp,ws,Rp,Rs);      % Gives mimimum order of filter
    [b,a]=ellip(n,Rp,Rs,wn);          % Elliptic filter design
elseif strcmp(STYLE,'FIR1')
    % Ap=1;
    % As=100;
    W1=(pl+sl)/2;
    W2=(ph+sh)/2;
    wdth=min((pl-sl),(sh-ph));
    N=ceil(11*pi/wdth)+1;
    b = fir1(N,[W1,W2]);
%     load b_2_23;
    a=1;
elseif strcmp(STYLE,'Ripple')
    %passband ripple of 0.01, stopband ripple of 0.1
    [n,Wn,bta,filtype]=kaiserord([fsl,fpl,fph,fsh],[0,1,0], [0.1,0.01,0.1],fs);
    b=fir1(n,Wn,filtype,kaiser(n+1,bta),'noscale');
    a=1;
else
    error('No valid filter style');
end
y=filter(b,a,x);
Hd = Bpf_baseShape1(fs,fsl,fpl,fph,fsh,Rs,Rp,Rs);
[b_lpfEnv,a_lpfEnv]=sos2tf(Hd.sosMatrix,Hd.ScaleValues);
% y=filter(b_lpfEnv,a_lpfEnv,x);
lb=floor(length(b)/2);
if strcmp(Fig,'on')
    [h,w]=freqz(b,a,256,fs);
    h=20*log10(abs(h));
    fprintf('%.3f,%.3f,%.3f,%.3f\n',ws,wp);
    figure;plot(w,h);title('bandpass filter');grid on;
end
end
