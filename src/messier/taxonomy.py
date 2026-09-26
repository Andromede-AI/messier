"""
Define the SOC and NAICS labels used for task classification.
"""

from typing import Literal
from pydantic import BaseModel, Field


# labels follow the 2018 SOC structure and 2022 NAICS sectors
SOC: dict[str, str] = {
    "11-1000": "Top Executives",
    "11-2000": "Advertising, Marketing, Promotions, Public Relations, and Sales Managers",
    "11-3000": "Operations Specialties Managers",
    "11-9000": "Other Management Occupations",
    "13-1000": "Business Operations Specialists",
    "13-2000": "Financial Specialists",
    "15-1200": "Computer Occupations",
    "15-2000": "Mathematical Science Occupations",
    "17-1000": "Architects, Surveyors, and Cartographers",
    "17-2000": "Engineers",
    "17-3000": "Drafters, Engineering Technicians, and Mapping Technicians",
    "19-1000": "Life Scientists",
    "19-2000": "Physical Scientists",
    "19-3000": "Social Scientists and Related Workers",
    "19-4000": "Life, Physical, and Social Science Technicians",
    "19-5000": "Occupational Health and Safety Specialists and Technicians",
    "21-1000": "Counselors, Social Workers, and Other Community and Social Service Specialists",
    "21-2000": "Religious Workers",
    "23-1000": "Lawyers, Judges, and Related Workers",
    "23-2000": "Legal Support Workers",
    "25-1000": "Postsecondary Teachers",
    "25-2000": "Preschool, Elementary, Middle, Secondary, and Special Education Teachers",
    "25-3000": "Other Teachers and Instructors",
    "25-4000": "Librarians, Curators, and Archivists",
    "25-9000": "Other Educational Instruction and Library Occupations",
    "27-1000": "Art and Design Workers",
    "27-2000": "Entertainers and Performers, Sports and Related Workers",
    "27-3000": "Media and Communication Workers",
    "27-4000": "Media and Communication Equipment Workers",
    "29-1000": "Healthcare Diagnosing or Treating Practitioners",
    "29-2000": "Health Technologists and Technicians",
    "29-9000": "Other Healthcare Practitioners and Technical Occupations",
    "31-1100": "Home Health and Personal Care Aides and Nursing Assistants, Orderlies, and Psychiatric Aides",
    "31-2000": "Occupational Therapy and Physical Therapist Assistants and Aides",
    "31-9000": "Other Healthcare Support Occupations",
    "33-1000": "Supervisors of Protective Service Workers",
    "33-2000": "Firefighting and Prevention Workers",
    "33-3000": "Law Enforcement Workers",
    "33-9000": "Other Protective Service Workers",
    "35-1000": "Supervisors of Food Preparation and Serving Workers",
    "35-2000": "Cooks and Food Preparation Workers",
    "35-3000": "Food and Beverage Serving Workers",
    "35-9000": "Other Food Preparation and Serving Related Workers",
    "37-1000": "Supervisors of Building and Grounds Cleaning and Maintenance Workers",
    "37-2000": "Building Cleaning and Pest Control Workers",
    "37-3000": "Grounds Maintenance Workers",
    "39-1000": "Supervisors of Personal Care and Service Workers",
    "39-2000": "Animal Care and Service Workers",
    "39-3000": "Entertainment Attendants and Related Workers",
    "39-4000": "Funeral Service Workers",
    "39-5000": "Personal Appearance Workers",
    "39-6000": "Baggage Porters, Bellhops, and Concierges",
    "39-7000": "Tour and Travel Guides",
    "39-9000": "Other Personal Care and Service Workers",
    "41-1000": "Supervisors of Sales Workers",
    "41-2000": "Retail Sales Workers",
    "41-3000": "Sales Representatives, Services",
    "41-4000": "Sales Representatives, Wholesale and Manufacturing",
    "41-9000": "Other Sales and Related Workers",
    "43-1000": "Supervisors of Office and Administrative Support Workers",
    "43-2000": "Communications Equipment Operators",
    "43-3000": "Financial Clerks",
    "43-4000": "Information and Record Clerks",
    "43-5000": "Material Recording, Scheduling, Dispatching, and Distributing Workers",
    "43-6000": "Secretaries and Administrative Assistants",
    "43-9000": "Other Office and Administrative Support Workers",
    "45-1000": "Supervisors of Farming, Fishing, and Forestry Workers",
    "45-2000": "Agricultural Workers",
    "45-3000": "Fishing and Hunting Workers",
    "45-4000": "Forest, Conservation, and Logging Workers",
    "47-1000": "Supervisors of Construction and Extraction Workers",
    "47-2000": "Construction Trades Workers",
    "47-3000": "Helpers, Construction Trades",
    "47-4000": "Other Construction and Related Workers",
    "47-5000": "Extraction Workers",
    "49-1000": "Supervisors of Installation, Maintenance, and Repair Workers",
    "49-2000": "Electrical and Electronic Equipment Mechanics, Installers, and Repairers",
    "49-3000": "Vehicle and Mobile Equipment Mechanics, Installers, and Repairers",
    "49-9000": "Other Installation, Maintenance, and Repair Occupations",
    "51-1000": "Supervisors of Production Workers",
    "51-2000": "Assemblers and Fabricators",
    "51-3000": "Food Processing Workers",
    "51-4000": "Metal Workers and Plastic Workers",
    "51-5100": "Printing Workers",
    "51-6000": "Textile, Apparel, and Furnishings Workers",
    "51-7000": "Woodworkers",
    "51-8000": "Plant and System Operators",
    "51-9000": "Other Production Occupations",
    "53-1000": "Supervisors of Transportation and Material Moving Workers",
    "53-2000": "Air Transportation Workers",
    "53-3000": "Motor Vehicle Operators",
    "53-4000": "Rail Transportation Workers",
    "53-5000": "Water Transportation Workers",
    "53-6000": "Other Transportation Workers",
    "53-7000": "Material Moving Workers",
    "55-1000": "Military Officer Special and Tactical Operations Leaders",
    "55-2000": "First-Line Enlisted Military Supervisors",
    "55-3000": "Military Enlisted Tactical Operations and Air/Weapons Specialists and Crew Members",
}

NAICS: dict[str, str] = {
    "11":    "Agriculture, Forestry, Fishing and Hunting",
    "21":    "Mining, Quarrying, and Oil and Gas Extraction",
    "22":    "Utilities",
    "23":    "Construction",
    "31-33": "Manufacturing",
    "42":    "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation and Warehousing",
    "51":    "Information",
    "52":    "Finance and Insurance",
    "53":    "Real Estate and Rental and Leasing",
    "54":    "Professional, Scientific, and Technical Services",
    "55":    "Management of Companies and Enterprises",
    "56":    "Administrative and Support and Waste Management and Remediation Services",
    "61":    "Educational Services",
    "62":    "Health Care and Social Assistance",
    "71":    "Arts, Entertainment, and Recreation",
    "72":    "Accommodation and Food Services",
    "81":    "Other Services except Public Administration",
    "92":    "Public Administration",
}

SocCode = Literal[*SOC]
NaicsCode = Literal[*NAICS]


class Classification(BaseModel):
    """
    Store one proposed SOC and NAICS classification.
    """

    soc_code: SocCode
    naics_code: NaicsCode
    rationale: str = Field(min_length=5)
    confidence: bool


VOTER_PROMPT = f"""You classify a task by SOC occupation code and NAICS sector code. Pick the single SOC code (the occupation whose work this task most resembles) and the single NAICS code (the industry where work of this kind would economically be performed - not where the exam or task itself originates).

SOC CODES (pick exactly one):
{chr(10).join(f"  {k}: {v}" for k, v in SOC.items())}

NAICS SECTORS (pick exactly one):
{chr(10).join(f"  {k}: {v}" for k, v in NAICS.items())}

Set confidence=false only if you cannot confidently place the task on either axis.

Output strictly as JSON:
  "soc_code":   one of the SOC codes above
  "naics_code": one of the NAICS codes above
  "rationale":  one sentence naming the occupation and the industry
  "confidence":  boolean"""

ADJUDICATOR_PROMPT = f"""You are the final adjudicator for split SOC + NAICS classifications. Three classifiers classified a task by occupation and industry, and they disagreed on at least one axis. Given the task and the three picks with rationales, decide the final SOC and NAICS.

You may pick from the three votes OR override all three if none captured the correct answer.

Principles:
- SOC = the human occupation whose work this task most resembles. Ask: 'If a person were doing this task, what is their job title?'
- NAICS = the industry where work of this kind would economically be performed (NOT where the exam or task itself originates).
- The benchmark format or test apparatus (e.g. a function-calling evaluation, a multi-turn dialog test, an exam) is NOT itself an industry. Classify based on the work the task simulates - a task simulating a flight reservation belongs in the transportation industry regardless of whether it's wrapped in an API benchmark.

SOC CODES (pick exactly one):
{chr(10).join(f"  {k}: {v}" for k, v in SOC.items())}

NAICS SECTORS (pick exactly one):
{chr(10).join(f"  {k}: {v}" for k, v in NAICS.items())}

Set confidence=false only if you cannot confidently place the task on either axis.

Output strictly as JSON:
  "soc_code":   one of the SOC codes above
  "naics_code": one of the NAICS codes above
  "rationale":  one sentence naming the occupation and the industry
  "confidence":  boolean"""
