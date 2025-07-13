import pandapower as pp
import numpy as np
import pandas as pd
import time

# Main system for generating the bus data
def generate_bus_data(data_str, system_name, n_scenarios=5000, min_sensors=5, max_sensors=10, 
                     noise_std_min=0.05, noise_std_max=0.1, missing_prob_min=0.3, missing_prob_max=0.5, 
                     load_scale_min=0.5, load_scale_max=1.5):
    """
    Generate simulated sensor data for an IEEE bus system from string data.
    
    Parameters:
    - data_str: String containing bus system data.
    - system_name: Name of the system (e.g., 'IEEE 33-bus', 'IEEE 69-bus').
    - n_scenarios: Number of scenarios to generate.
    - min_sensors, max_sensors: Range for number of sensors per scenario.
    - noise_std_min, noise_std_max: Range for Gaussian noise standard deviation.
    - missing_prob_min, missing_prob_max: Range for missing data probability.
    - load_scale_min, load_scale_max: Range for load scaling.
    
    Outputs:
    - CSV files: sensor_inputs_<system_name>.csv, grid_states_<system_name>.csv
    """
    # Parse data from string
    try:
        lines = [line.strip().split() for line in data_str.strip().split('\n') if line.strip()]
        data = [[int(line[0])-1, int(line[1])-1, float(line[2])/1000, float(line[3])/1000, float(line[4]), float(line[5]), float(line[6])] for line in lines]
    except Exception as e:
        print(f"Error parsing data for {system_name}: {e}")
        return

    # Determine number of buses
    n_buses = max(max(line[0], line[1]) for line in data) + 1

    # Create pandapower network
    net = pp.create_empty_network()
    buses = [pp.create_bus(net, vn_kv=12.66, name=f"Bus_{i}") for i in range(n_buses)]
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0, name="Slack")

    # Create lines and loads
    for from_bus, to_bus, p_mw, q_mvar, r_ohm, x_ohm, max_i_a in data:
        pp.create_line_from_parameters(net, from_bus=buses[from_bus], to_bus=buses[to_bus], length_km=1, r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm, c_nf_per_km=0, max_i_ka=max_i_a/1000, name=f"Line_{from_bus}_{to_bus}")
        if p_mw > 0 or q_mvar > 0:  # Load at to_bus
            pp.create_load(net, bus=buses[to_bus], p_mw=p_mw, q_mvar=q_mvar, name=f"Load_{to_bus}")

    # Initialize data storage
    input_data = []
    output_data = []
    bus_indices = list(range(n_buses))
    line_indices = [(i, j) for i, j in net.line[['from_bus', 'to_bus']].values]

    # Generate scenarios
    np.random.seed(42)  # For reproducibility
    start_time = time.time()
    for scenario in range(n_scenarios):
        print(f"Generating scenario {scenario + 1}/{n_scenarios} for {system_name}...")
        
        # Randomly scale loads
        for idx in net.load.index:
            scale = np.random.uniform(load_scale_min, load_scale_max)
            net.load.at[idx, 'p_mw'] *= scale
            net.load.at[idx, 'q_mvar'] *= scale

        # Run load flow
        try:
            pp.runpp(net, max_iteration=100)
        except pp.LoadflowNotConverged:
            print(f"Load flow failed for scenario {scenario + 1}. Skipping...")
            continue

        # Get true grid states (outputs)
        vm_pu = net.res_bus.vm_pu.values
        va_degree = net.res_bus.va_degree.values
        output_row = np.concatenate([vm_pu, va_degree])

        # Randomly select number of sensors
        n_sensors = np.random.randint(min_sensors, max_sensors + 1)
        sensor_buses = np.random.choice(bus_indices, size=n_sensors // 2, replace=False)
        sensor_lines = np.random.choice(len(line_indices), size=n_sensors - len(sensor_buses), replace=False)
        
        # Initialize input row
        input_row = []
        measurement_types = []
        
        # Voltage measurements
        for bus in sensor_buses:
            true_val = net.res_bus.vm_pu[bus]
            noise_std = np.random.uniform(noise_std_min, noise_std_max)
            if np.random.uniform() > np.random.uniform(missing_prob_min, missing_prob_max):
                val = true_val * (1 + np.random.normal(0, noise_std))
            else:
                val = np.nan
            input_row.append(val)
            measurement_types.append(f'V_bus_{bus}')
        
        # Current and power flow measurements
        for line_idx in sensor_lines:
            line = net.res_line.iloc[line_idx]
            true_i = line.i_ka * 1000  # Convert kA to A
            true_p = line.p_from_mw
            true_q = line.q_from_mvar
            noise_std = np.random.uniform(noise_std_min, noise_std_max)
            missing_prob = np.random.uniform(missing_prob_min, missing_prob_max)
            
            # Current
            if np.random.uniform() > missing_prob:
                i_val = true_i * (1 + np.random.normal(0, noise_std))
            else:
                i_val = np.nan
            input_row.append(i_val)
            measurement_types.append(f'I_line_{line_idx}')
            
            # Active power
            if np.random.uniform() > missing_prob:
                p_val = true_p * (1 + np.random.normal(0, noise_std))
            else:
                p_val = np.nan
            input_row.append(p_val)
            measurement_types.append(f'P_line_{line_idx}')
            
            # Reactive power
            if np.random.uniform() > missing_prob:
                q_val = true_q * (1 + np.random.normal(0, noise_std))
            else:
                q_val = np.nan
            input_row.append(q_val)
            measurement_types.append(f'Q_line_{line_idx}')

        # Store data
        input_data.append(input_row)
        output_data.append(output_row)

    # Create DataFrames
    input_columns = [f'meas_{i}' for i in range(len(input_row))]
    output_columns = [f'V_{i}' for i in range(n_buses)] + [f'theta_{i}' for i in range(n_buses)]
    input_df = pd.DataFrame(input_data, columns=input_columns)
    output_df = pd.DataFrame(output_data, columns=output_columns)

    # Save to CSV
    system_name_clean = system_name.lower().replace(" ", "_")
    input_df.to_csv(f'sensor_inputs_{system_name_clean}.csv', index=False)
    output_df.to_csv(f'grid_states_{system_name_clean}.csv', index=False)

    # Print summary
    print(f"\nData generation for {system_name} complete in {time.time() - start_time:.2f} seconds.")
    print(f"Input data shape: {input_df.shape}")
    print(f"Output data shape: {output_df.shape}")
    print(f"Sparsity: {input_df.isna().mean().mean():.2%}")
    print(f"Measurement types for last scenario: {measurement_types}")
