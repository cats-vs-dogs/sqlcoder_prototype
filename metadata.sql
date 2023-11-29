CREATE TABLE Customers (
	STICHTAG , --The [Reporting date] refers to the economic date to which the represented data apply. This means that the data record was obtained from the operational business systems on that date after the close of online activities and bookings. Format 'YYYYMMDD'. 
	PARTNER_ID , --Client specific unique identifier of a partner. 
	KZ_ZAHLUNGSVERZUG , --Relevant information in the STA approaches: If the Flag_Past_Due = 'J' then the partner/transaction is in default.Reference: Article 127 CRR 
	PD_FINAL , --Final PD of the transaction component. The final PD is either the punctual PD ([PD punctual transaction component including PD add-ons]) or the PD class as derived via rule table [RT].[Class PD] 
	SL_RISIKOLAND , --Risk country of the risk-partner of the transaction - without taking into account any collaterals.(corresponds to the ISO-Alpha-2 country code). 
	RATINGVERFAHREN_ID , --The Rating procedure_ID facilitates a client-specific unique and detailed identification of all rating/scoring procedures used. This is necessary for a targeted controlling and analysis in the area of ratings/scorings. The ID is chosen by the client itself (6 characters).   In HVB AG, the ID consists of the rating type and the number of the special case from the rating system KRIBS Rating. Together with the information on the version number and the score card ID from the reporting information, it is therefore possible to tell for every entry in the rating entity precisely what rating/scoring procedure, what version and what score card was used to estimate the relevant PD. 
	SL_NACE , --The NACE-Code (Statistical Classification of Economic Activities in the European Community) is the industry standard classification system used in the European Union and gives the economic activity of the customer/ company. 
	BETR_UNTERNEHMENSGROESSE , --Relevant information in the IRB approaches: The amount company size is used to distinguish between the asset classes corporate large and corporate small and is one input to calculate the risk weight for small corporates. The amount company size is filled according to the following rule: Turnover (if available) or Balance sheet total (if available) or Annual incomeReference: Article 153, 4 CRR 
);

CREATE TABLE Collaterals (
	STICHTAG , --The [Reporting date] refers to the economic date to which the represented data apply. This means that the data record was obtained from the operational business systems on that date after the close of online activities and bookings. Format 'YYYYMMDD'. 
	SICHERHEITENVEREINB_ID_LGD_SI , --Client specific unique identifier of a LGD collateral.Necessary to uniquely identify a LGD collateral (Primary keys: Collateral agreement ID LGD collateral, Sub-collateral ID LGD collateral, Transaction category collateral LGD collateral). 
	MARKTWERT , --Current market value of a derivative/ SFT transaction. 
);

CREATE TABLE Transactions (
	STICHTAG , --The [Reporting date] refers to the economic date to which the represented data apply. This means that the data record was obtained from the operational business systems on that date after the close of online activities and bookings. Format 'YYYYMMDD'. 
	GESCHAEFT_ID , --Client specific unique identifier of a transaction. 
	PARTNER_ID , --Client specific unique identifier of a partner. 
	SICHERHEITENVEREINB_ID_LGD_SI , --Client specific unique identifier of a LGD collateral.Necessary to uniquely identify a LGD collateral (Primary keys: Collateral agreement ID LGD collateral, Sub-collateral ID LGD collateral, Transaction category collateral LGD collateral). 
	EXPOSURE_CLASS_TRANSACTION , --Exposure class of the transaction - without taking into account the adjustments due to collateralisation.Reference: Article 112/ 147 CRR 
	SL_VERARBEITUNG , --Internal processing key used by the calculation engine to cluster transactions based on their product category (e.g. KREDIT - on balance sheet credit transaction, AUSSER - off-balance sheet credit transaction, FORDPA - debt securities, etc.) 
	KATEGORIE_SICHERHEIT_LGD_SI , --This key clusters the different LGD collaterals (Financial collaterals, residential real estate, commercial real estate, receivables, credit protection, etc.) 
	INFRASTRUCTURE_SUPPORT_FACTOR , --nan 
	SME_FACTOR , --SME supporting factor determined in accordance with Article 501 CRR. (The SME supporting factor reduces the RWA of transactions towards small and medium-sized enterprises.) 
	CORRELATION_MULTIPLIER , --Correlation multiplier used in the RW formula for Banks/ Corporates. Possible values: 1,25 or 1. 
	RESTLAUFZEIT , --Residual maturity of the transaction  in days. The residual maturity is calculated in CE as [Date: End of transaction]  [Reporting date] 
	URSPRUNGSWAEHRUNG , --Gives the original currency (ISO code) of the transaction 
	URSPRUNGSLAUFZEIT , --Gives the original maturity of the transaction which is calculated as [Date: End of transaction] - [Date: Start of transaction] 
	DAT_GESCGAEFTSENDE , --NO DESCRIPTION WAS FOUND 
	HOPT , --NO DESCRIPTION WAS FOUND 
	ORIG_EXPOSURE_PRE_CONV_FACTOR , --Basis for assessment for COREP purposes- before value adjustment/provisions (STC)- before EAD parameter (LEQ, LOF)- before EAD collateral - before weighting factor product- after netting effects 
	EAD , --Exposure at default (final) 
	LGD_GESCHAEFTSBESTANDTEIL , --LGD (including downturn effects)of the transactions component considering existing collaterals (LGD collaterals + substitution collaterals) 
	RISIKOAKTIVA , --Amount of the risk-weighted assets per transaction component 
	ERWARTETER_VERLUST , --(Expected Loss) shows the absolute amount of the expected loss for the transaction component under observation. 
);

