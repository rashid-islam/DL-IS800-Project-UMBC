clear, clc
avg_HR_ae_ncf = dlmread('avg_HR_ae_ncf.txt');
avg_HR_iae = dlmread('avg_HR_iae.txt');
avg_HR_NCF = dlmread('avg_HR_NCF.txt');
avg_HR_uae = dlmread('avg_HR_uae.txt');

avg_NDCG_ae_ncf = dlmread('avg_NDCG_ae_ncf.txt');
avg_NDCG_iae = dlmread('avg_NDCG_iae.txt');
avg_NDCG_NCF = dlmread('avg_NDCG_NCF.txt');
avg_NDCG_uae = dlmread('avg_NDCG_uae.txt');


K = (1:10)';

figure(1)
plot(K,avg_HR_iae,'-m*','LineWidth',3,'MarkerSize',7,'MarkerIndices',1:2:10)
hold on
plot(K,avg_HR_ae_ncf,'-kp','LineWidth',3,'MarkerSize',7,'MarkerIndices',2:3:10)
plot(K,avg_HR_uae,'-go','LineWidth',3,'MarkerSize',7,'MarkerIndices',1:2:10)
plot(K,avg_HR_NCF,'-b+','LineWidth',3,'MarkerSize',7,'MarkerIndices',2:3:10)
xlim([1 10])
%ylim([0 0.75])
legend('IAE','AE-NCF','UAE','NCF')
%legend({'NCF w/ Pre-train','NFCF','NFCF\_embd','NCF w/o Pre-train','DNN Classifier','MF w/ Pre-train','MF w/o Pre-train','BPMF'},'NumColumns',2);
xlabel('K')
ylabel('HR@K')
set(gca,'fontsize',17)
set(gca,'fontname','times')
set(gca,'XTick',1:1:10)
%legend boxoff  

figure(2)
plot(K,avg_NDCG_iae,'-m*','LineWidth',3,'MarkerSize',7,'MarkerIndices',1:2:10)
hold on
plot(K,avg_NDCG_ae_ncf,'-kp','LineWidth',3,'MarkerSize',7,'MarkerIndices',2:3:10)
plot(K,avg_NDCG_uae,'-go','LineWidth',3,'MarkerSize',7,'MarkerIndices',1:2:10)
plot(K,avg_NDCG_NCF,'-b+','LineWidth',3,'MarkerSize',7,'MarkerIndices',2:3:10)
xlim([1 10])
%ylim([0 0.75])
legend('IAE','AE-NCF','UAE','NCF')
%legend({'NCF w/ Pre-train','NFCF','NFCF\_embd','NCF w/o Pre-train','DNN Classifier','MF w/ Pre-train','MF w/o Pre-train','BPMF'},'NumColumns',2);
xlabel('K')
ylabel('NDCG@K')
set(gca,'fontsize',17)
set(gca,'fontname','times')
set(gca,'XTick',1:1:10)


