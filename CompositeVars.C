#include "TROOT.h"
#include "TH1.h"
#include "TTree.h"
#include "TH2.h"
#include "TF1.h"
#include "TFile.h"
#include "TMath.h"
#include "TSystem.h"

using namespace std;

using std::cout;
using std::endl;

void CompositeVars(){

   //Open input files
   TFile *file1(0);
   TFile *file2(0);
   TString fnameData = "/lstore/cms/boletti/ntuples/2018Data_passPreselection_passSPlotCuts_mergeSweights.root";
   TString fnameMC = "/lstore/cms/boletti/ntuples/MC_JPSI_2018_preBDT_Nov21.root";

   if (!gSystem->AccessPathName( fnameData )) {
      file1 = TFile::Open( fnameData ); // check if file in directory exists
   }
   if (!file1) {
      std::cout << "ERROR: could not open Data file" << std::endl;
      exit(1);
   }

   if (!gSystem->AccessPathName( fnameMC )) {
      file2 = TFile::Open( fnameMC ); // check if file in directory exists
   }
   if (!file2) {
      std::cout << "ERROR: could not open MC file" << std::endl;
      exit(1);
   }


   //Open TTree of input files
   TTree* dataTree = (TTree*) file1->Get("ntuple");
   TTree* mcTree = (TTree*) file2->Get("ntuple");

   //Create two output ROOT files
   TFile * fout1= new TFile("Data2018_CompositeVars.root","RECREATE");
   TFile * fout2= new TFile("MC_JPSI_2018_CompositeVars.root","RECREATE");

    
   //Declare old variables
   Double_t blbs;
   Double_t blbsE;
   Double_t bCos;
   Double_t bCosE;

   //Declare new variables
   Double_t ratio;
   Double_t ratioCos;


    
   //Output 1: Data_CompositeVars.root
   ////////////////////////////////////////////////////////////////////////////////////////////
   fout1->cd();

   //Cloning TTree
   //TTree* dataTree_2 = dataTree->CopyTree("tagged_mass < 5.14 || tagged_mass > 5.41");
   TTree* dataTree_2 = dataTree->CloneTree(-1);

   dataTree->SetBranchAddress("bLBS", &blbs);
   dataTree->SetBranchAddress("bLBSE", &blbsE);
   dataTree->SetBranchAddress("bCosAlphaBS", &bCos);
   dataTree->SetBranchAddress("bCosAlphaBSE", &bCosE);

   TBranch* nv1 = dataTree_2->Branch("bLBSRatio", &ratio);
   TBranch* nv2 = dataTree_2->Branch("bCosRatio", &ratioCos);

   //Fill TBranches
   for (int i = 0; i < dataTree->GetEntries(); i++) {
   dataTree->GetEntry(i);

   ratio = blbs / blbsE;
   nv1->Fill();
   
   ratioCos = (1 - bCos) / bCosE;
   nv2->Fill();
   }

   dataTree_2->Write();
   fout1->Close();



   //Output 2: MC_CompositeVars.root
   ////////////////////////////////////////////////////////////////////////////////////////////
   fout2->cd();

   //Cloning TTree
   //TTree* mcTree_2 = mcTree->CopyTree("tagged_mass > 5.14 && tagged_mass < 5.41");
   TTree* mcTree_2 = mcTree->CloneTree(-1);

   mcTree->SetBranchAddress("bLBS", &blbs);
   mcTree->SetBranchAddress("bLBSE", &blbsE);
   mcTree->SetBranchAddress("bCosAlphaBS", &bCos);
   mcTree->SetBranchAddress("bCosAlphaBSE", &bCosE);

   TBranch* nv1MC = mcTree_2->Branch("bLBSRatio", &ratio);
   TBranch* nv2MC = mcTree_2->Branch("bCosRatio", &ratioCos);

   //Fill TBranches
   for (int i = 0; i < mcTree->GetEntries(); i++) {
   mcTree->GetEntry(i);

   ratio = blbs / blbsE;
   nv1MC->Fill();
   
   ratioCos = (1 - bCos) / bCosE;
   nv2MC->Fill();
   }


   mcTree_2->Write();
   fout2->Close();



   //Close input files
   file1->Close();
   file2->Close();

   return;
   
}