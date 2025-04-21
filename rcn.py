building_data = {
    "Rumah Sederhana": 2978318,
    "Rumah Menengah": 4775925,
    "Rumah Mewah": 7030198,
    "Ruko": 3670422,
    "Hotel Budget": 6282211,
    "Hotel Bintang 3": 7110014,
    "Hotel Bintang 4": 7841097,
    "Hotel Bintang 5": 10038122,
    "Hotel Bintang 5+": 10808618,
    "Kantor Rise C": 6535577,
    "Kantor Rise D": 7614644,
    "Kantor Rise A": 8411664,
    "Apartemen Luxury": 9923974 + 9497461,  # Summing the two values for Apartemen Luxury
    "Apartemen Mid-Up": 7872975,
    "Apartemen Mid-Mid": 7172630,
    "Apartemen Mid-Low": 6163124,
    "Mall Mid-High": 5084207,
    "Mall Mid-Up": 5585820,
    "Mall Mid-Low": 4716148,
    "Gudang": 3010214,
    "Kanopi": 542322,
    "Rumah Sakit": 6313237,
    "Gedung Logistik": 2749881,
    "Khazanah BNI": 12081500,
    "Semi Permanen": 1499775,
    "Pembangkit": 10772309
}




estate_data = depreciation_rates = {
    "Rumah Sederhana": {"Estate": 0.30, "Non Kawasan": 0.10, "building_age": 20},
    "Rumah Menengah": {"Estate": 0.25, "Non Kawasan": 0.10, "building_age": 30},
    "Rumah Mewah": {"Estate": 0.20, "Non Kawasan": 0.10, "building_age": 40},
    "Ruko": {"Estate": 0.15, "Non Kawasan": 0.15, "building_age": 30},
    "Hotel Budget": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Hotel Bintang 3": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Hotel Bintang 4": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Hotel Bintang 5": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Hotel Bintang 5+": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Kantor Rise C": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Kantor Rise B": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Kantor Rise A": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Kantor Rise A+": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Apartemen Luxury": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Apartemen Mid-Up": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Apartemen Mid-Mid": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Apartemen Mid-Low": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Mall Mid-Mid": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 40},
    "Mall Mid-Up": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 40},
    "Mall Mid-Low": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 40},
    "Gudang": {"Estate": 0.15, "Non Kawasan": 0.10, "building_age": 30},
    "Kanopi": {"Estate": 0.0, "Non Kawasan": 0.0, "building_age": 20},
    "Rumah Sakit": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Gudang Logistik": {"Estate": 0.20, "Non Kawasan": 0.20, "building_age": 50},
    "Khazanah BNI": {"Estate": 0.10, "Non Kawasan": 0.10, "building_age": 50},
    "Semi Permanen": {"Estate": 0.10, "Non Kawasan": 0.10, "building_age": 30},
    "Pembangkit": {"Estate": 0.15, "Non Kawasan": 0.15, "building_age": 30}
}

condition_format = {
    "New": 100,
    "Very Good": 85,
    "Good": 70,
    "Adequate": 50,
    "Poor": 35,
    "Scrap": 10
}




def format_rupiah(amount):
    """
    Formats a numeric amount (integer or float) into a string representing the amount in Indonesian Rupiah format.
    
    Parameters:
    amount (int or float): The numeric amount of money to be formatted.
    
    Returns:
    str: The formatted string in Indonesian Rupiah format (e.g., 'Rp1.000.000,00').
    
    Raises:
    TypeError: If the input is not a numeric type (int or float).
    ValueError: If the input is a negative number.
    """
    # Check if the input is a numeric type (int or float)
    if not isinstance(amount, (int, float)):
        raise TypeError("Input must be an integer or float.")
    
    # Check if the input is a negative number
    if amount < 0:
        raise ValueError("Input must be a non-negative number.")
    
    # Round the amount to 2 decimal places
    # rounded_amount = round(amount, 2)
        
    # Format with commas as thousand separators and replace commas with periods for thousands,
    # then replace dot for decimals with comma for Indonesian style
    formatted_amount = "{:,.2f}".format(amount).replace(",", "X").replace(".", ",").replace("X", ".")
    
    # Add 'Rp' prefix
    rupiah_format = f"Rp{formatted_amount}"
    
    return rupiah_format



