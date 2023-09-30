from ModelHandler import *

feature_names = [
    ### Good
    "trkValidFrac",
    "glbTrackProbability",
    "nLostHitsInner",
    "nLostHitsOuter",
    "trkKink",
    "chi2LocalPosition",
    "match2_dX",
    "match2_pullX",
    "match1_dX",
    "match1_pullX",

    ### Weak but useful
    "nPixels",
    "nValidHits",
    "nLostHitsOn",
    "match2_dY",
    "eta",
    "match1_dY",
    "match2_pullY",
    "match1_pullY",
    "match2_pullDyDz",
    "match1_pullDyDz",
    "match2_pullDxDz",
    "match1_pullDxDz",
    "pt",
]

class MuonMVA(ModelHandler):

    def get_true_classification(self, data):
        return (data["sim_type"] != 1)

    def get_parameters(self):
        parameters = super(MuonMVA, self).get_parameters()
        parameters['min_child_weight'] = 0.01
        parameters['max_depth'] = 5
        parameters['eta'] = 0.3
        parameters['tree_method'] = 'hist' 
        return parameters

def study_model(files, name):
    
    model = MuonMVA("results/study_muon_mva", feature_names, "muons", "evt")
    model.do_weight=True
    model.load_datasets(files)
    model.apply_selection(model.data["pt"]>2)
  #  model.apply_selection(model.data["isPF"])
   # model.apply_selection(abs(model.data["eta"])<1.4)

    number_of_splits = 3
    train_model_for_all_splits = False
    for event_index in range(number_of_splits):
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        print("Number of signal/background events in training sample: %u/%u (%0.3f)" \
            % (sum(model.y_train == True),
               sum(model.y_train == False),
               sum(model.y_train == True) / float(sum(model.y_train == False))))
        model_name = "%s-%s-Event%u" % (name, date, event_index)
        #model.add_weight()
        model_files=["results/weight_muon_mva_pt3/Run2022-20230815-2359-Event0.model",\
                     "results/unweighted_muon_mva_pt2/Run2022-20230816-2342-Event0.model",\
                     "results/unweighted_muon_mva_addpt_pt2/Run2022-20230817-2132-Event0.model",\
                     "results/weight_muon_mva_addpt_pt2/Run2022-20230816-0513-Event0.model",\
                     "results/weight_muon_mva_pt2/Run2022-20230816-1535-Event0.model"] 
        label_names=["weighted training pt>3","unweighted training pt>2","unweighted training (add pt) pt>2","weighted training (add pt) pt>2","weighted training pt>2"]

        model.load_cat_file()
        model.load_file(model_files,label_names)

        if not train_model_for_all_splits:
            break

def check_model(files, name):
    model = MuonMVA("results/study_muon_mva", feature_names, "muons", "evt")
    model.do_weight = False
    model.load_datasets(files)
    model.make_kinematic_plot()
    
    model = MuonMVA("results/study_muon_mva", feature_names, "muons", "evt")
    model.data = None
    model.do_weight = True   
    model.load_datasets(files)
    model.make_kinematic_plot()

if __name__ == "__main__":
    date = datetime.now().strftime("%Y%m%d-%H%M")

    #data_path = "/eos/cms/store/group/phys_bphys/bmm/bmm6/PostProcessing/FlatNtuples/524/muon_mva/"
    data_path = "/eos/user/w/wangz/tem/524/muon_mva/" 
    files_Run2022 = [
        data_path + "InclusiveDileptonMinBias_TuneCP5Plus_13p6TeV_pythia8+Run3Summer22MiniAODv3-Pilot_124X_mcRun3_2022_realistic_v12-v5+MINIAODSIM/"
    ]
    study_model(files_Run2022, "Run2022")
  #  check_model(files_Run2022, "Run2022")
