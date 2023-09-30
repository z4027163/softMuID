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

def train_model(files, name):
    
    model = MuonMVA("results/muon_mva", feature_names, "muons", "evt")
    model.do_weight=True
    model.load_datasets(files)    
    model.apply_selection(model.data["pt"]>2)
    #model.apply_selection(model.data["pt"]<6)
    #model.apply_selection(abs(model.data["eta"])<1.4)

    number_of_splits = 3
    train_model_for_all_splits = False
    temdata=model.data
    print(model.data)
    for event_index in range(number_of_splits):
        model.data=temdata
        print("bin1")
        print(model.data)
        model.apply_selection(model.data["pt"]<4)
        model.apply_selection(abs(model.data["eta"])<1.4)
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        model_name = "%s-bin11-Event%u" % (name, event_index)
        model.train(model_name)
        
        model.data=temdata
        model.apply_selection(model.data["pt"]>=4)
        model.apply_selection(model.data["pt"]<6)
        model.apply_selection(abs(model.data["eta"])<1.4)
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        model_name = "%s-bin21-Event%u" % (name, event_index)
        model.train(model_name)

        model.data=temdata
        model.apply_selection(model.data["pt"]>=6)
        model.apply_selection(model.data["pt"]<8)
        model.apply_selection(abs(model.data["eta"])<1.4)
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        model_name = "%s-bin31-Event%u" % (name, event_index)
        model.train(model_name)

        model.data=temdata
        model.apply_selection(model.data["pt"]>=8)
        model.apply_selection(model.data["pt"]<10)
        model.apply_selection(abs(model.data["eta"])<1.4)
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        model_name = "%s-bin41-Event%u" % (name, event_index)
        model.train(model_name)

        model.data=temdata
        model.apply_selection(model.data["pt"]>=10)
        model.apply_selection(abs(model.data["eta"])<1.4)
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        model_name = "%s-bin51-Event%u" % (name, event_index)
        model.train(model_name)

        model.data=temdata
        model.apply_selection(model.data["pt"]<4)
        model.apply_selection(abs(model.data["eta"])>=1.4)
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        model_name = "%s-bin12-Event%u" % (name, event_index)
        model.train(model_name)

        model.data=temdata
        model.apply_selection(model.data["pt"]>=4)
        model.apply_selection(model.data["pt"]<6)
        model.apply_selection(abs(model.data["eta"])>=1.4)
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        model_name = "%s-bin22-Event%u" % (name, event_index)
        model.train(model_name)

        model.data=temdata
        model.apply_selection(model.data["pt"]>=6)
        model.apply_selection(model.data["pt"]<8)
        model.apply_selection(abs(model.data["eta"])>=1.4)
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        model_name = "%s-bin32-Event%u" % (name, event_index)
        model.train(model_name)

        model.data=temdata
        model.apply_selection(model.data["pt"]>=8)
        model.apply_selection(model.data["pt"]<10)
        model.apply_selection(abs(model.data["eta"])>=1.4)
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        model_name = "%s-bin42-Event%u" % (name, event_index)
        model.train(model_name)

        model.data=temdata
        model.apply_selection(model.data["pt"]>=10)
        model.apply_selection(abs(model.data["eta"])>=1.4)
        model.prepare_train_and_test_datasets(number_of_splits, event_index)
        model_name = "%s-bin52-Event%u" % (name, event_index)
        model.train(model_name)

        if not train_model_for_all_splits:
            break

if __name__ == "__main__":
    date = datetime.now().strftime("%Y%m%d-%H%M")

    data_path = "/eos/cms/store/group/phys_bphys/bmm/bmm6/PostProcessing/FlatNtuples/524/muon_mva/"
    #data_path = "/eos/user/w/wangz/tem/524/muon_mva/" 
    #files_Run2022 = data_path + "InclusiveDileptonMinBias_TuneCP5Plus_13p6TeV_pythia8+Run3Summer22MiniAODv3-Pilot_124X_mcRun3_2022_realistic_v12-v5+MINIAODSIM/" 
    files_Run2022 = [
        data_path + "InclusiveDileptonMinBias_TuneCP5Plus_13p6TeV_pythia8+Run3Summer22MiniAODv3-Pilot_124X_mcRun3_2022_realistic_v12-v5+MINIAODSIM/"
    ]
    train_model(files_Run2022, "Run2022")


