function [max_acc] = run_kmeans_acc_from_python(class, subject_factor)
    [acc, sen, spec, d, nmi, fscore, pp] = return_kmeans_additionalmetrics(class', subject_factor');

    max_acc = max(acc(:))/100;
end
