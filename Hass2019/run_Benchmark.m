output_folder = pwd;
copyfile(fullfile(pwd, 'arFit.m'),fullfile(pwd,'d2d','arFramework3','arFit.m'))
cd(fullfile(pwd,'d2d','arFramework3'))
arInit
d2d_folder = pwd;


for model = {'Boehm_JProteomeRes2014','Brannmark_JBC2010', ...
             'Crauste_ImmuneCells_CellSystems2017', 'Fiedler_BMC2016', ...
             'Fujita_SciSignal2010', 'Weber_BMC2015', 'Zheng_PNAS2012'}
    cd(fullfile(d2d_folder, 'Examples', model{1}))
    
    if strcmp(model{1}, 'Boehm_JProteomeRes2014')
        Setup_FullModel_Boehm2014
    elseif strcmp(model{1},'Zheng_PNAS2012')
        Setup_Zheng
    else
        Setup
    end
    if strcmp(model{1}, 'Fujita_SciSignal2010')
        ar.config.fiterrors = 0;
    end
    % lsqnonlin
    ar.config.optimizer = 1;
    ar.config.atol = 1e-8;
    ar.config.rtol = 1e-8;
    ar.config.useFitErrorCorrection = 0;
    ar.config.optim.TolX = 1e-6;
    ar.config.optim.TolFun = 0;
    ar.config.optim.PreconBandWidth = Inf;
    ar.config.optim.SubproblemAlgorithm = 'factorization';
    ar.config.optim.MaxIter = 1e5;
    ar.config.optim.MaxFunEvals = 1e5;
    mat_savefile = fullfile(output_folder, strcat(model{1}, '_lsqnonlin.mat'));
    if isfile(mat_savefile)
        load(mat_savefile)
    else
        arFitLHS(1000, 0)
        save(mat_savefile, 'ar')
    end
    for field = {'iter', 'chi2s', 'ps'}
        dlmwrite(fullfile(output_folder, strcat(model{1}, '_lsqnonlin_',field{1},'.csv')), ar.(field{1}), 'delimiter', ',', 'precision', 12)
    end
    
    % fmincon
    ar.config.optimizer = 2;
    ar.config.atol = 1e-8;
    ar.config.rtol = 1e-8;
    ar.config.useFitErrorCorrection = 0;
    ar.config.optim.TolX = 1e-6;
    ar.config.optim.TolFun = 0;
    ar.config.optim.PreconBandWidth = Inf;
    ar.config.optim.MaxIter = 1e5;
    ar.config.optim.MaxFunEvals = 1e5;
    
    mat_savefile = fullfile(output_folder, strcat(model{1}, '_fmincon.mat'));
    if isfile(mat_savefile)
        load(mat_savefile)
    else
        arFitLHS(1000, 0)
        save(mat_savefile, 'ar')
    end
    for field = {'iter', 'chi2s', 'ps'}
        dlmwrite(fullfile(output_folder, strcat(model{1}, '_fmincon_',field{1},'.csv')), ar.(field{1}), 'delimiter', ',', 'precision', 12)
    end
    
    cd(d2d_folder)
end

cd(output_folder);