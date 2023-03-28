# Hotel Booking Exploration with Causal Inference

**Task:** Estimating the impact of assigning a different room as compared to what the customer had reserved on Booking Cancellations.

The dataset contains booking information for a city hotel and a resort hotel taken from a real hotel in Portugal, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things. All personally identifying information has been removed from the data.

There are 2 datasets representing 2 hotels. One of the hotels (H1) is a resort hotel and the other is a city hotel (H2). Both datasets share the same structure, with 31 variables describing the 40,060 observations of H1 and 79,330 observations of H2. Each observation represents a hotel booking.

This repository contains a detailed data exploration process along with a full causal inference cycle implemented in the dowhy library. Besides, different algorithms for causal estimation are also implemented from scratch in simple versions for educational purpose.

---
**References:**

Antonio et. al. (2019). *Hotel booking demand datasets*. Data in Brief, Volume 22. [DOI](https://doi.org/10.1016/j.dib.2018.11.126)

Siddharth Dixit. (2020). *Beyond Predictive Models: The Causal Story Behind Hotel Booking Cancellations*. [Towards Data Science](https://towardsdatascience.com/beyond-predictive-models-the-causal-story-behind-hotel-booking-cancellations-d29e8558cbaf)
