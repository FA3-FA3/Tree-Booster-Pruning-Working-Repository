import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Example mapping of filter names to central wavelengths (in nanometers)
filter_wavelengths = {
    "GAIA/GAIA3.Gbp": 532, "GAIA/GAIA3.G": 673, "GAIA/GAIA3.Grp": 797,
    "SLOAN/SDSS.u": 355, "SLOAN/SDSS.g": 468, "SLOAN/SDSS.r": 616,
    "SLOAN/SDSS.i": 748, "SLOAN/SDSS.z": 893,
    "2MASS/2MASS.J": 1250, "2MASS/2MASS.H": 1650, "2MASS/2MASS.Ks": 2150,
    "WISE/WISE.W1": 3350, "WISE/WISE.W2": 4600, "WISE/WISE.W3": 11500, "WISE/WISE.W4": 22000,
}

def extract_photometric_data(sample):
    """
    Extracts photometric filter values from a multi-index pandas Series.
    :param sample: A row of the DataFrame (df.iloc[row])
    :return: Dictionary {filter_name: intensity}
    """
    photometric_data = {}

    for (category, filter_name, value_type, *_), intensity in sample.items():
        if category == "Model" and value_type == "Value":  # Ensure it's a photometric value
            if pd.notna(intensity):  # Ignore NaN values
                photometric_data[filter_name] = intensity  # Store intensity in Jy

    return photometric_data

def plot_photometric_graph(sample):
    """
    Extracts and plots a photometric graph from a sample row.
    :param sample: A row from df.iloc[row], containing photometric values.
    """
    sample_data = extract_photometric_data(sample)  # Extract photometry
    wavelengths = []
    intensities = []

    for filter_name, intensity in sample_data.items():
        if filter_name in filter_wavelengths:
            wavelengths.append(filter_wavelengths[filter_name])  # Get wavelength
            intensities.append(intensity)  # Get intensity in Jy

    # Convert to numpy arrays and sort
    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities)

    if len(wavelengths) == 0:
        print("No valid photometric data available for plotting.")
        return

    sorted_indices = np.argsort(wavelengths)
    wavelengths = wavelengths[sorted_indices]
    intensities = intensities[sorted_indices]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(wavelengths, intensities, marker="o", linestyle="-", color="b", label="Flux (Jy)")
    
    plt.xscale("log")  # Log scale for better wavelength spacing
    plt.yscale("log")  # Log scale for intensity
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (Jy)")
    plt.title("Photometric Spectrum")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()