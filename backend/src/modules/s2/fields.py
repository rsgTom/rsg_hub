# fields.py

# Graph API fields
GRAPH_API_PAPER_FIELDS = [
    "paperId", "corpusId", "url", "title", "venue", "publicationVenue", "year",
    "authors", "externalIds", "abstract", "referenceCount", "citationCount",
    "influentialCitationCount", "isOpenAccess", "openAccessPdf", "fieldsOfStudy",
    "s2FieldsOfStudy", "publicationTypes", "publicationDate", "journal",
    "citationStyles", "embedding.specter_v1", "embedding.specter_v2", "tldr"
]

GRAPH_API_AUTHOR_FIELDS = [
    "authorId", "externalIds", "url", "name", "affiliations", "homepage",
    "paperCount", "citationCount", "hIndex"
]

GRAPH_API_CITATION_FIELDS = [
    "contexts", "intents", "contextsWithIntent", "isInfluential", "paperId",
    "corpusId", "url", "title", "venue", "publicationVenue", "year", "authors",
    "externalIds", "abstract", "referenceCount", "citationCount", "influentialCitationCount",
    "isOpenAccess", "openAccessPdf", "fieldsOfStudy", "s2FieldsOfStudy", "publicationTypes",
    "publicationDate", "journal", "citationStyles"
]

GRAPH_API_REFERENCE_FIELDS = [
    "paperId", "corpusId", "url", "title", "venue", "publicationVenue", "year",
    "authors", "externalIds", "abstract", "referenceCount", "citationCount",
    "influentialCitationCount", "isOpenAccess", "openAccessPdf", "fieldsOfStudy",
    "s2FieldsOfStudy", "publicationTypes", "publicationDate", "journal", "citationStyles"
]

# Recommendations API fields
RECOMMENDATIONS_API_FIELDS = [
    "paperId", "corpusId", "url", "title", "venue", "publicationVenue", "year",
    "authors", "externalIds", "abstract", "referenceCount", "citationCount",
    "influentialCitationCount", "isOpenAccess", "openAccessPdf", "fieldsOfStudy",
    "s2FieldsOfStudy", "publicationTypes", "publicationDate", "journal", "citationStyles"
]

# Datasets API fields (add fields as needed)
DATASETS_API_FIELDS = [
    # Add specific dataset fields here if available
]
