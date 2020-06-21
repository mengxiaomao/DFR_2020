clc;clear; close all;

len=10e3;
npacket=5e2;
band = 15; %MHz
Band = band *1e6;
SNR_range = 10; % [0,10]dB
A=1;
B=1;
rg=0;
M=4;
K=2;

fc=70e6;
fi=70e6;
samrate=4;

Npre_1=100;
Nsuf_1=10;

Npre=1;
Nsuf=2;
NL=Npre+Nsuf+1;
sam=M*K*samrate;
Len=sam*NL;
    
train_x=zeros(len,Len);
train_y=zeros(len,M);
for k=1:len/npacket
    %for band-limited system
    j=sqrt(-1);

    N=M*K;
    fs=fc*samrate;
    deltat=1/fs;
    
    RB=fc/N;
    T=1/RB;%symbol duration

    %% s[k] series after modulation
    logM=log2(M);
    xbit=randi([0,1],npacket*logM,1);
    xsym=bi2de(reshape(xbit,logM,npacket).','left-msb');%[1,M]
    xsym_temp=[zeros(Npre_1,1);xsym;zeros(Nsuf_1,1)];
%             xsym=[zeros(Npre_1,1);xsym;zeros(Nsuf_1,1)];

    sl = mppsk_bb_modulator1(xsym_temp+1,A,B,K,N,rg,1);
    sl = kron(sl,ones(1,samrate));
%     s_ini=[zeros(1,sam*Npre_1),sl,zeros(1,sam*Nsuf_1)];

    %% carrier
    t=0:deltat:T*(Npre_1+Nsuf_1+npacket)-deltat;
    carrier=exp(j*2*pi*fc*t);
    s=real(sl.*carrier);
%     [Pxx,F] =pwelch(s,[],[],[],fs,'centered');Pxx=10*log10(Pxx);
    
    apass=3;
    astop=45;
    [s,lb]=bandp1(s,fi-Band/2,fi+Band/2,fi-Band/2-Band/10,fi+Band/2+Band/10,apass,astop,fs,'cheby2','off');%

%     figure;p1=plot(F,Pxx-max(Pxx));hold on;
%     [Pxx,F] =pwelch(real(s),[],[],[],fs,'centered');Pxx=10*log10(Pxx);
%     p2=plot(F,Pxx-max(Pxx),'r--');xlabel('Frequency (Hz)');ylabel('Amplitude (dB)');grid on;legend('s(t)','x(t)');
%     set(p1,'Linewidth',1); %line 1 property
%     set(p2,'Linewidth',1); % line 2 property

    %% additional noise
    SNR=rand()*SNR_range;
    coff=1/10^(-SNR/10);
    maxp=sum(s.*s)/length(s);
    Pn=maxp/coff;
    s=s+sqrt(Pn)*randn(size(s));
    [s,lb]=bandp1(s,fi-Band/2,fi+Band/2,fi-Band/2-Band/10,fi+Band/2+Band/10,apass,astop,fs,'cheby2','off');%
%     maxind=ceil(lb/2);
%     s=s(maxind+1:end);
    
    indfirst=1:npacket;
    indfirst=indfirst+Npre_1-Npre-1;
    indlast=indfirst+NL-1;
    tra_x=zeros(npacket,Len);
    tra_y=zeros(npacket,M);
    for i=1:npacket
        ind=indfirst(i);
        tempf=ind*sam+1;
        templ=tempf+Len-1;

        tempdata=s(tempf:templ);
        tra_x(i,:)=tempdata;
        tra_y(i,xsym(i)+1)=1;
    end
    train_x((k-1)*npacket+1:k*npacket,:)=tra_x;
    train_y((k-1)*npacket+1:k*npacket,:)=tra_y;
end

train_data=[train_x,train_y];
train_data=single(train_data);
FILENAME=strcat('train_data_',num2str(band),'MHz_train');
save(FILENAME,'train_data');