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

void check_roc(){

  TFile *root_file = new TFile("results/weight_muon_mva_pt4/histogram_low-weight.root");
  TH1F *sig_hist = (TH1F*)root_file->Get("sig");
  TH1F *bkg_hist = (TH1F*)root_file->Get("bkg");

  sig_hist->Scale(1/sig_hist->Integral());
  bkg_hist->Scale(1/bkg_hist->Integral());


  TFile *root_file2 = new TFile("results/weight_muon_mva_pt4/histogram_hi-weight.root");

  TH1F *sig_hist2 = (TH1F*)root_file2->Get("sig");
  TH1F *bkg_hist2 = (TH1F*)root_file2->Get("bkg");

  sig_hist2->Scale(1/sig_hist2->Integral());
  bkg_hist2->Scale(1/bkg_hist2->Integral());

  int binx=sig_hist->GetNbinsX();

  TH1F *roc_1 = new TH1F("roc1","roc1",binx,0,1);
  TH1F *roc_2 = new TH1F("roc2","rco2",binx,0,1);

  double a1=0.2;
  double b1=0.075;
  double a2=0.08;
  double x1[50];
  double y1[50];
  double x2[50];
  double y2[50];
  for(int i=1;i<=binx;i++){
     double e_sig1= sig_hist->Integral(i,binx)*(1-a1)+sig_hist2->Integral(i,binx)*a1;
     double e_bkg1= bkg_hist->Integral(i,binx)*(1-b1)+bkg_hist2->Integral(i,binx)*b1;
     x1[i-1]=e_bkg1;
     y1[i-1]=e_sig1;
     double e_sig2= sig_hist->Integral(i,binx)*(1-a2)+sig_hist2->Integral(i,binx)*a2;
     double e_bkg2= bkg_hist->Integral(i,binx)*(1-a2)+bkg_hist2->Integral(i,binx)*a2;
     x2[i-1]=e_bkg2;
     y2[i-1]=e_sig2;
  }
  auto g1 = new TGraph(binx,x1,y1);
  auto g2 = new TGraph(binx,x2,y2);

  TCanvas *c = new TCanvas();
  c->cd();
  g1->SetLineColor(2);
  g1->SetLineWidth(3);
  g2->SetLineColor(4);
  g2->SetLineWidth(3);

 // g1->Draw("AC");
 //  g2->Draw("AC same");



  TLegend *leg = new TLegend(0.4,0.5,0.9,0.7);
  leg->AddEntry(g1,"pt>8 sig frac =0.2, bgk frac=0.075","l");
  leg->AddEntry(g2,"pt>8 common frac =0.08","l");
  leg->SetBorderSize(0);

  TMultiGraph *mg = new TMultiGraph();
  mg->Add(g1);
  mg->Add(g2);
  mg->Draw("AC");
  leg->Draw("same");
  mg->GetYaxis()->SetTitle("sig eff");
  mg->GetXaxis()->SetTitle("bkg eff");

  c->SaveAs("roc_study-weight2.png");
}
