# Copyright (c) Microsoft. All rights reserved.

from dataclasses import dataclass
from typing import Optional, List
from pydantic import BaseModel, Field

###
# The data model used for this sample is based on the hotel data model from the Azure AI Search samples.
# When deploying a new index in Azure AI Search using the import wizard you can choose to deploy the 'hotel-samples'
# dataset, see here: https://learn.microsoft.com/en-us/azure/search/search-get-started-portal.
# This is the dataset used in this sample with some modifications.
# This model adds vectors for the 2 descriptions in English and French.
# Both are based on the 1536 dimensions of the OpenAI models.
# You can adjust this at creation time and then make the change below as well.
###


@dataclass
class EmployeeHandbookModel(BaseModel):
    """Data model for employee handbook documents with vector embeddings."""
    
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    parent_id: Optional[str] = Field(None, description="Parent document ID")
    content: str = Field(..., description="Text content of the chunk")
    title: str = Field(..., description="Title of the document")
    url: str = Field(..., description="URL of the document")
    filepath: str = Field(..., description="File path of the document")
    contentVector: Optional[List[float]] = Field(
        None, 
        description="Vector embedding of the content (1536 dimensions for OpenAI)"
    )

   