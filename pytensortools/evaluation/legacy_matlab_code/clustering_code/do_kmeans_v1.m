function [score, flabels, dist] = do_kmeans_v1(ClassLabel, A, nb)
%removed the replication step since it can be implicitly done by kmeans
%function
N       = length(unique(ClassLabel));
% opts    = statset('Display','final');
% [labels,~,d] = kmeans(A,N,'Replicates', nb,'Options',opts);
[labels,~,d]   = kmeans(A,N,'Replicates', nb);
score   = 0;
P       = perms(1:N);
for j = 1:size(P,1)
     plabels{j} = P(j,labels)';
     ps = sum(ClassLabel == plabels{j});
    if (ps > score)
        score  = ps;            
        index = j;            
    end
end
flabels = plabels{index};
dist    = sum(d);

