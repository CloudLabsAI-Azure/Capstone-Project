from typing import TypedDict, Annotated, Optional  
import requests  
import asyncio  
import os
from dotenv import load_dotenv

load_dotenv(override=True)

class GeoPlugin:  
    """Plugin for geocoding locations to latitude and longitude."""
    
    async def get_latitude_longitude(self, location: str) -> str:
        """
        Gets the latitude and longitude for a location.
        
        Args:
            location: The name of the location
            
        Returns:
            String containing latitude and longitude
        """
        print(f"lat/long request location: {location}")
        url = f"https://geocode.maps.co/search?q={location}&api_key={os.getenv('GEOCODING_API_KEY')}"  
        response = requests.get(url) 
        data = response.json() 
        position = data[0]
        return f"Latitude: {position['lat']}, Longitude: {position['lon']}"
    