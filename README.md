# Random number simulation of the Dunning and Kruger experiments

The Dunning–Kruger effect is a cognitive bias wherein people with limited competence in a particular domain overestimate their abilities. It turns out that the Dunning and Kruger experiments do not prove that the effect is real. This app illustrates this using a random number simulation. The data generation process doesn't imply that "incompetent" participants overestimate their abilities. See more details in my blog post [Debunking the Dunning–Kruger effect with random number simulation](https://e10v.me/debunking-dunning-kruger-effect/).

The app is available in the Streamlit Community Cloud: https://dunning-kruger.streamlit.app

To install and run it locally, follow these steps:

0. Install [PDM](https://pdm.fming.dev/latest/#installation).

1. Download the app code and go to the directory:

```bash
git clone git@github.com:e10v/dunning-kruger.git
cd dunning-kruger
```

2. Install dependencies:

```bash
pdm sync
```

3. Run the app:

```bash
pdm run app
```

4. Open the app: http://localhost:8501
