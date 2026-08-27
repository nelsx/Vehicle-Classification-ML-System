import numpy as np
import joblib
from pathlib import Path

CURRENT_DIR = Path.cwd()

def get_valid_input(prompt, type_func, min_val=None):
    while True:
        try:
            user_input=input(prompt).strip()
            if user_input.lower()=='quit':
                return quit

            value=type_func(user_input)

            if min_val is not None and value<min_val:
                print(f"Value must be at least {min_val}. Please try again.")
                continue

            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {type_func.__name__} number.")

def predictive_system():
    print("==================================================")
    print("   Initializing Vehicle Classification System     ")
    print("==================================================")

    try:
        model=joblib.load(CURRENT_DIR.parent/'Artifacts'/'Logistic_Regression_Model.joblib')
        scaler=joblib.load( CURRENT_DIR.parent/'Artifacts'/'StandardScaler_Logistic_Regression.joblib')
        encoder=joblib.load(CURRENT_DIR.parent/'Artifacts'/'LabelEncoder_Fuel_Type.joblib')
        print("Model, Scaler, and Encoder loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading system files: {e}")
        print("Please ensure your .joblib files exist in your working directory.")
        return


    while True:
        print("\n Enter vehicle details below (or type 'quit' to exit):")
        print("-"*50)

        CMPG = get_valid_input("Enter City MPG: ", int, 0)
        if CMPG == 'quit': break

        HMPG = get_valid_input("Enter Highway MPG: ", int, 0)
        if HMPG == 'quit': break

        CCMPG = get_valid_input("Enter Combined MPG: ", int, 0)
        if CCMPG == 'quit': break

        Engine_Cylinders = get_valid_input("Enter Engine Cylinders : ", float, 0.0)
        if Engine_Cylinders == 'quit': break

        Engine_Size_L = get_valid_input("Enter Engine Size in Liters : ", float, 0.0)
        if Engine_Size_L == 'quit': break

        CO2_Emissions = get_valid_input("Enter CO2 Emissions (g/mile): ", int, 0)
        if CO2_Emissions == 'quit': break

        EV_Range= get_valid_input("Enter EV Range in miles (0 for non-EVs): ", int, 0)
        if EV_Range == 'quit': break


        denominator = CCMPG if CCMPG >0 else 1

        FE = (CMPG + HMPG + CCMPG) / 3
        PP = Engine_Cylinders * Engine_Size_L
        CO2_MPG = CO2_Emissions / denominator

        raw_features=[Engine_Cylinders, Engine_Size_L, CMPG, HMPG, CCMPG, CO2_Emissions, EV_Range, FE, PP, CO2_MPG]

        input_array=np.array(raw_features).reshape(1, -1)
        scaled_features=scaler.transform(input_array)
        prediction=model.predict(scaled_features)

        print("\n" + "=" * 40)
        if prediction[0] == 1:
            print("The vehicle is classified as an Electric Vehicle.")
        elif prediction[0] == 2:
            print("The vehicle is classified as a Hybrid Vehicle.")
        elif prediction[0] == 3:
            print("The vehicle is classified a Diesel Vehicle.")
        else:
            print("The vehicle is classified as a Petroleum Vehicle.")

        print("=" * 40 + "\n")

        again = input("Evaluate another vehicle? (y/n): ").strip().lower()
        if again != 'y':
            break

    print("\nClosing classification interface. Goodbye!")

if __name__ == "__main__":
    predictive_system()