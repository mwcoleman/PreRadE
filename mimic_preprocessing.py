#### From https://github.com/Shaumik-Ashraf/BART-MIMIC-CXR/ #####

import os
import pandas as pd
import re

ROOT = os.path.dirname( os.path.abspath(__file__) )
LIST_FILE = os.path.join(ROOT, 'data', 'cxr-study-list.csv.gz')
METADATA_FILE = os.path.join(ROOT, 'data', 'mimic-cxr-2.0.0-metadata.csv.gz')
REPORTS_DIR = os.path.join(ROOT, 'data', 'mimic-cxr-reports')
REPORTS_OUTPUT = os.path.join(ROOT, 'data', 'reports_clean_impressions.csv')

# Additional file that links dicom_id to reports
REPORTS_WITH_DID_OUTPUT = os.path.join(ROOT, 'data', 'id_to_findings.csv')


def remove_notification_section(text):
	"""
	We noticed that some reports have a notification section after
	the impressions (summary) section, which was impeding our data, so
	we decided to remove this section all together. We use various rule-based
	mechanisms to parse and remove the notification section.
	params: text
	returns: text with notification section removed
	"""
	idx = text.rfind("NOTIFICATION")
	if( idx > 0 ):
		text = text[:idx]
	idx = text.rfind("telephone notification")
	if( idx > 0 ):
		text = text[:idx]
	idx = text.rfind("Telephone notification")
	if( idx > 0 ):
		text = text[:idx]
	idx = text.rfind("These findings were")
	if( idx > 0 ):
		text = text[:idx]
	idx = text.rfind("Findings discussed")
	if( idx > 0 ):
		text = text[:idx]
	idx = text.rfind("Findings were")
	if( idx > 0 ):
		text = text[:idx]
	idx = text.rfind("This preliminary report")
	if( idx > 0 ):
		text = text[:idx]
	idx = text.rfind("Reviewed with")
	if( idx > 0 ):
		text = text[:idx]
	idx = text.rfind("A preliminary read")
	if( idx > 0 ):
		text = text[:idx]
	return(text)

def parse_summary(text):
	"""
	parses and separates input text from summary in cxr reports, returns None if
	not found
	params: text
	returns: None or [input_text, summary]
	"""

	regex = r'impression.?(?::|" ")'

	if( re.search(regex, text, flags=re.IGNORECASE)==None ): #if no summary
		return None

	data = re.split(regex, text, flags=re.IGNORECASE)
	data[0] = data[0].strip()
	data[1] = data[1].strip()

	return(data)

def write_csv(filename, reports):
	"""
	writes a csv file for summarization. The CSV file has four columns: "subject_id",
	"study_id", "findings", and "impression" based on MIMIC-CXR reports. "findings"
	contains the input text, and "impression" contains the true summary.
	params: filename - name of csv file to write, will overwrite if it exists
		reports - dataframe of cxr reports from cxr-study-list file
	"""
	print(f"Writing {filename}...")
	f = open(filename, 'w')
	f.write(f"\"subject_id\",\"study_id\",\"findings\",\"impression\"\n")
	ommitted = 0
	progress = 1
	for report in reports:
		x = open(os.path.join(REPORTS_DIR, report))
		text = x.read()
		x.close()
		text = sanitize(text)
		if( text==None ):
			ommitted += 1
			# continue; #toss out data and go to next textfile

		if (progress % 10000 == 0):
			print(f'Read {progress} files so far...')
		progress += 1

		data = parse_summary(text)
		# data[0]: findings, data[1]: impression
		# TODO: swap comment if want to filter on impressions (not findings)
		if( (data==None) or (data[0]=='') or (data[1]=='') ):
		# if( (data==None) or (data[1]=='') ):
			ommitted += 1
			continue; #toss out data and go to next textfile

		folders = report.split('/')
		f.write(f"\"{folders[2]}\",\"{folders[3].split('.')[0]}\",\"{data[0]}\",\"{data[1]}\"\n")
	f.close()
	print(f"Ommited {ommitted} files out of {progress} total files in dataset.\n")
	print("Done.\n")

def sanitize(text):
	"""
	Cleanses the text to be written in CSV, which will be fed directly to
	the summarizer. Tokenization and lemmatization is not performed in this
	step, as the summarizer performs those directly.
	params: text
	returns: cleaned text
	"""
	text = text.strip()
	text = re.sub("\n", "", text)
	text = re.sub(",", "", text)
	# Remove all text before FINDINGS: section
	regex = r'^(.*finding.?:)'
	
	# TODO: comment if want to filter on impressions (not findings)
	if( re.search(regex, text, flags=re.IGNORECASE)==None ): #if no summary
		return None

	text = re.sub(regex,"", text, flags=re.IGNORECASE)
	text = remove_notification_section(text)
	return(text)


def merge_filter_records(out_file, reports_clean=REPORTS_OUTPUT, meta_data=METADATA_FILE):
	"""Produces DF with dicom_id, study_id, findings, impressions, + others
	(for those reports that have findings only)"""
	metadata = pd.read_csv(meta_data)
	reports_clean = pd.read_csv(reports_clean)

	# Strip leading s and convert to str
	reports_clean.loc[:,'study_id'] = reports_clean.study_id.apply(lambda x: int(x[1:]))

	valid_sids = [int(v[1:]) for v in reports_clean['study_id']]
	txt_data = metadata[metadata['study_id'].isin(valid_sids)].copy()
	txt_data.drop(columns=['Rows', 'Columns', 'StudyDate', 'StudyTime',
						   'PerformedProcedureStepDescription', 
						   'ProcedureCodeSequence_CodeMeaning', 
						   'ViewCodeSequence_CodeMeaning', 
						   'PatientOrientationCodeSequence_CodeMeaning', 
						   'filepath'], inplace=True)
	
	txt_data = txt_data.merge(reports_clean, on='study_id', how='right')
	txt_data.rename(columns={'subject_id_x':'subject_id'}, inplace=True)

	txt_data.to_csv(REPORTS_WITH_DID_OUTPUT)

if __name__=='__main__':
	print("================ Starting data preprocessing ==================")

	print(f"Reading {os.path.basename(LIST_FILE)}...")
	rad_reports = pd.read_csv(LIST_FILE)['path'] # file paths as pandas series
	print("Done.")

	write_csv(REPORTS_OUTPUT, rad_reports)

	### MC: Merge and filter metadata
	out_file = input('If you wish to create linked dicom_id to report csv, enter filename (otherwise blank):')
	if out_file != '':
		merge_filter_records(out_file, reports_clean=REPORTS_OUTPUT, meta_data=METADATA_FILE)

	print("==================== End data preprocessing ======================")