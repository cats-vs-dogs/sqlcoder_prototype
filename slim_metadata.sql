CREaTE TABLE Transactions (
	ORIG_EXPOSURE_PRE_CONV_FACTOR REAL --Basis for assessment for COREP purposes- before value adjustment/provisions (STC)- before EAD parameter (LEQ
	INFRASTRUCTURE_SUPPORT_FACTOR REAL --nan
	EAD REAL --Exposure at default (final)
	PARTNER_ID TEXT --Client specific unique identifier of a partner.
)

