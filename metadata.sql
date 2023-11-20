CREATE TABLE portfolio (
  REPORTING_DATE DATE -- Reporting date
  PARTNER_ID VARCHAR(50) -- ID of the partner
  LOAN_ID VARCHAR(50) -- ID of the loadn
  PRODUCT VARCHAR(50) -- The loan product
  STATUS VARCHAR(50) -- Status of the loan. Either "Performing" or "Non performing"
  SEGMENT VARCHAR(50) -- Segment of the client business
  EXPOSURE_DRAWN NUMERIC -- Drawn exposure
  EXPOSURE_UNDRAWN NUMERIC -- Undrawn exposure
  EXPOSURE NUMERIC -- Exposure
  EAD NUMERIC -- Exposure at default
  EAD_COLL NUMERIC -- Exposure at default
  PD NUMERIC -- Probability of default
  LGD NUMERIC -- Loss given default
  SIZE NUMERIC -- The size of tje client in MEUR, usually this is the client's turnover
  MATURITY NUMERIC -- Maturity of the load
  F_MORTGAGE CHAR(1)  -- If 'Y' the loan is a mortgage, else 'N'
  F_REVOLVING CHAR(1) -- If 'Y' the loan is revolving, else 'N'
  F_LARGE_FIN CHAR(1) -- If 'Y' the client is large financial institution, else 'N'
  RW NUMERIC -- Risk weight
  RWA NUMERIC -- Risk weight average
);

