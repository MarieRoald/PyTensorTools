function [acc, sen, spec, d, nmi, fscore, pp] = return_kmeans_additionalmetrics(t, data)
%patient:1, control:2

R = size(data,2);
b = 1;
for i=1:R
    temp = combnk(1:R,i);
    for j=1:size(temp,1)
        [score, nlabels,dist] = do_kmeans_v1(t, data(:,temp(j,:)), 100);
        % accuracy
        TP = length(find(nlabels==1 & t==1));
        TN = length(find(nlabels==2 & t==2));
        FP = length(find(nlabels==1 & t==2));
        FN = length(find(nlabels==2 & t==1));
        sen(i,j)    = TP/ (TP+FN)*100;
        spec(i,j)   = TN/ (TN +FP)*100;
        acc(i,j)    = score/length(t)*100;
        d(i,j)      = dist;       
        pp(i,j) = 1/length(t)*(length(find(nlabels(t==1)==1))+length(find(nlabels(t==2)==2))); %purity
        p = TP/(TP+FP);
        r = TP/(TP+FN);
        fscore(i,j) =  ((b^2+1)*p*r)/(b^2*p+r); %f score
        nmi(i,j) = compute_nmi(nlabels, t);     %normalized mutual information (nmi)
    end
end