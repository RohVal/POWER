from requests import get

# should be in a .env file
API_KEY = "91a5427b58214288825135502242006"

base_url = "http://api.weatherapi.com/v1"
# base_url/history.json/{api_key}&{location}&{date}
full_url = "https://api.weatherapi.com/v1/history.json?key=91a5427b58214288825135502242006&q=52.4020, -0.94378&dt=2024-06-19"


def make_request(date: str) -> dict:
    """
    Make a request to the weather API.

    Args:
        date (str): The date to make the request for. Format: 'YYYY-MM-DD'.

    Returns:
        dict: The response from the API.
    """
    response = get(f"{base_url}/history.json?key={API_KEY}&q=52.4020, -0.94378&dt={date}")
    return response.json()


def parse_response(response: dict) -> dict:
    """
    Parse the response from the API.

    Args:
        response (dict): The response from the API.

    Returns:
        dict: The parsed response.
    """

    data = {
        repr(hour): {
            "temp": response["forecast"]["forecastday"][0]["hour"][hour]["temp_c"],
            "wind": response["forecast"]["forecastday"][0]["hour"][hour]["wind_kph"],
            # we are assuming that the wind_max is the gust
            "wind_max": response["forecast"]["forecastday"][0]["hour"][hour]["gust_kph"],
            "wind_dir": response["forecast"]["forecastday"][0]["hour"][hour]["wind_dir"],
        } for hour in range(24)
    }

    return data

if __name__ == "__main__":
    res = make_request("2024-06-19")

    data = parse_response(res)
    print(data)
