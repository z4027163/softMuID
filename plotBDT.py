import ROOT
import matplotlib.pyplot as plt

# Open the ROOT file
root_file = ROOT.TFile("results/weight_muon_mva_addpt_pt2/histogram_low.root")

# Get the histograms from the ROOT file
sig_hist = root_file.Get("sig")
bkg_hist = root_file.Get("bkg")

sig_hist.Scale(1/sig_hist.Integral())
bkg_hist.Scale(1/bkg_hist.Integral())


root_file2 = ROOT.TFile("results/weight_muon_mva_addpt_pt2/histogram_hi.root")

sig_hist2 = root_file2.Get("sig")
bkg_hist2 = root_file2.Get("bkg")

sig_hist2.Scale(1/sig_hist2.Integral())
bkg_hist2.Scale(1/bkg_hist2.Integral())

ROOT.gStyle.SetOptStat(0)

canvas = ROOT.TCanvas("canvas", "Histogram Canvas")

sig_hist.SetTitle("xgboost")
#sig_hist.SetStats(kFALSE);
sig_hist.Draw("HIST")
sig_hist.SetMaximum(0.4)
sig_hist.SetLineColor(ROOT.kRed)
sig_hist.SetLineWidth(3)
bkg_hist.Draw("HIST SAME")
bkg_hist.SetLineColor(ROOT.kBlue)
bkg_hist.SetLineWidth(3)

#sig_hist2.SetStats(kFALSE);
sig_hist2.Draw("HIST same")
sig_hist2.SetLineColor(6)
sig_hist2.SetLineStyle(2)
sig_hist2.SetLineWidth(3)
bkg_hist2.Draw("HIST SAME")
bkg_hist2.SetLineColor(7)
bkg_hist2.SetLineStyle(2)
bkg_hist2.SetLineWidth(3)

legend = ROOT.TLegend(0.2, 0.65, 0.4, 0.85)
legend.AddEntry(sig_hist, "Signal (4<pt<8)", "l")
legend.AddEntry(bkg_hist, "Background (4<pt<8)", "l")
legend.AddEntry(sig_hist2, "Signal (pt>8)", "l")
legend.AddEntry(bkg_hist2, "Background (pt>8)", "l")

legend.SetBorderSize(0)
legend.Draw()
canvas.Update()
canvas.SaveAs("plot_hist-pt2-weight.png")

root_file.Close()
root_file2.Close()
