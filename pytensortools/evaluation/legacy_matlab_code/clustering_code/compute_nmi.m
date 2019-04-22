function nmi = compute_nmi(nlabels, class)

K = length(unique(nlabels));
J = length(unique(class));
N = length(nlabels);
for i=1:K
    w(i) = length(find(nlabels==i));
end
for j=1:J
    c(j) = length(find(class==j));
end
%entropy
Hw = 0;
Hc = 0;
for i=1:K
    Hw  = Hw + (- w(i)/N * log2(w(i)/N));
end
for i=1:J
    Hc  = Hc + (- c(i)/N * log2(c(i)/N));
end
%mutual info
I =0;
for k=1:K
    for j=1:J
        inter = length(find(class(nlabels==k)==j));
        if inter==0
            I=I;
        else
            I = I + (inter/N)*log2((N*inter)/(w(k)*c(j)));
        end
    end
end
nmi = I/((Hw+Hc)/2);

