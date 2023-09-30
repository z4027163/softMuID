#include <TMath.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include <math.h>
#include <stdlib.h>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdlib>
#include <stdio.h>

#include <TGraph.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TROOT.h>

void check_525(){

  TFile *root_file = new TFile("/eos/user/w/wangz/tem/525/muon_mva/InclusiveDileptonMinBias_TuneCP5Plus_13p6TeV_pythia8+Run3Summer22MiniAODv3-Pilot_124X_mcRun3_2022_realistic_v12-v5+MINIAODSIM/merge.root");
  TTree *t = (TTree*)root_file->Get("muons");


  TString var[4]={"trkValidFrac","glbTrackProbability","trkKink","chi2LocalPosition"};
  
  TString binning[4]={"(50,0,1)","(80,0,40)","(50,0,200)","(50,0,50)"};

  int i=1;
 
  t->Draw(var[i]+">>h1"+binning[i],"sim_type!=1");
  TH1F *h1d = (TH1F*)gDirectory->Get("h1");
   
  t->Draw(var[i]+">>h2"+binning[i],"isGlobal && sim_type!=1");
  TH1F *h2d = (TH1F*)gDirectory->Get("h2");

  t->Draw(var[i]+">>h3"+binning[i],"sim_type==1");
  TH1F *h3d = (TH1F*)gDirectory->Get("h3");

  t->Draw(var[i]+">>h4"+binning[i],"isGlobal && sim_type==1");
  TH1F *h4d = (TH1F*)gDirectory->Get("h4");

  h1d->Scale(1/h1d->Integral());
  h2d->Scale(1/h2d->Integral());
  h3d->Scale(1/h3d->Integral());
  h4d->Scale(1/h4d->Integral());
  h1d->SetStats(kFALSE);
  h3d->SetStats(kFALSE);
  h1d->SetMaximum(h1d->GetMaximum()*1.5);
  TCanvas *c = new TCanvas();
  c->cd();
  h1d->SetLineColor(2);
  h1d->SetLineWidth(3);
  h2d->SetLineColor(4);
  h2d->SetLineWidth(3);
  h3d->SetLineColor(6);
  h3d->SetLineWidth(3);
  h4d->SetLineColor(7);
  h4d->SetLineWidth(3);


  TLegend *leg = new TLegend(0.4,0.5,0.6,0.7);
  leg->AddEntry(h1d,"sig","l");
  leg->AddEntry(h2d,"sig (isGlobal)","l");
  leg->AddEntry(h3d,"bkg","l");
  leg->AddEntry(h4d,"bkg (isGlobal)","l");
  leg->SetBorderSize(0);

  h1d->Draw("HIST");
  h2d->Draw("HIST same");
  h3d->Draw("HIST same");
  h4d->Draw("HIST same");
  leg->Draw("same");
  c->SaveAs("var525_"+var[i]+".png");
}
