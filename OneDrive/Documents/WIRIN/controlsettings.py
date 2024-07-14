from flask import Flask, jsonify, request
import threading

app = Flask(__name__)

# Define the status dictionary to keep track of the states
status = {
    "ControlSettings": {
        "LowLevelControlMode": "Manual Mode",
        "PIDStatus": {
            "MasterControl": "OFF",
            "SteeringRack": "OFF",
            "Brake": "OFF",
            "Motors": "OFF"
        },
        "MasterPIDValues": {
            "SteeringPIDOutput": 0,
            "BrakePIDOutput": 0,
            "MotorRPIDOutput": 0,
            "MotorLPIDOutput": 0,
            "MasterPIDCommandOutput": 0
        }
    }
}

# Define threading functions for each component
def control_settings_thread():
    @app.route('/controlsettings/lowlevelcontrolmode/<mode>', methods=['POST'])
    def set_low_level_control_mode(mode):
        if mode in ["Autonomous LEVEL 5", "Autonomous LEVEL 4", "Autonomous LEVEL 3", "Autonomous LEVEL 2", "Autonomous LEVEL 1", "Manual Mode", "Hardware Mode"]:
            status["ControlSettings"]["LowLevelControlMode"] = mode
            return jsonify({"status": f"Low Level Control Mode is now {mode}"}), 200
        return jsonify({"error": "Invalid mode"}), 400

    @app.route('/controlsettings/lowlevelcontrolmode', methods=['GET'])
    def get_low_level_control_mode():
        return jsonify({"LowLevelControlMode": status["ControlSettings"]["LowLevelControlMode"]}), 200

def pid_status_thread():
    @app.route('/controlsettings/pidstatus/<component>/<action>', methods=['POST'])
    def set_pid_status(component, action):
        if component in status["ControlSettings"]["PIDStatus"] and action in ["ON", "OFF"]:
            status["ControlSettings"]["PIDStatus"][component] = action
            return jsonify({"status": f"{component} is now {action}"}), 200
        return jsonify({"error": "Invalid component or action"}), 400

    @app.route('/controlsettings/pidstatus/<component>', methods=['GET'])
    def get_pid_status(component):
        if component in status["ControlSettings"]["PIDStatus"]:
            return jsonify({component: status["ControlSettings"]["PIDStatus"][component]}), 200
        return jsonify({"error": "Invalid component"}), 400

def master_pid_values_thread():
    @app.route('/controlsettings/masterpidvalues/<component>/<int:value>', methods=['POST'])
    def set_master_pid_values(component, value):
        if component in status["ControlSettings"]["MasterPIDValues"]:
            if component in ["SteeringPIDOutput", "BrakePIDOutput"]:
                if -1024 <= value <= 1024:
                    status["ControlSettings"]["MasterPIDValues"][component] = value
                    return jsonify({"status": f"{component} is now {value}"}), 200
            elif component in ["MotorRPIDOutput", "MotorLPIDOutput"]:
                if 0 <= value <= 5000:
                    status["ControlSettings"]["MasterPIDValues"][component] = value
                    return jsonify({"status": f"{component} is now {value}"}), 200
            elif component == "MasterPIDCommandOutput":
                if 0 <= value <= 1000:
                    status["ControlSettings"]["MasterPIDValues"][component] = value
                    return jsonify({"status": f"{component} is now {value}"}), 200
        return jsonify({"error": "Invalid component or value"}), 400

    @app.route('/controlsettings/masterpidvalues/<component>', methods=['GET'])
    def get_master_pid_values(component):
        if component in status["ControlSettings"]["MasterPIDValues"]:
            return jsonify({component: status["ControlSettings"]["MasterPIDValues"][component]}), 200
        return jsonify({"error": "Invalid component"}), 400

# Define endpoints for retrieving the status of each component
@app.route('/status/controlsettings', methods=['GET'])
def get_control_settings_status():
    return jsonify(status["ControlSettings"]), 200

# Start the threads
threads = []
threads.append(threading.Thread(target=control_settings_thread))
threads.append(threading.Thread(target=pid_status_thread))
threads.append(threading.Thread(target=master_pid_values_thread))

for thread in threads:
    thread.start()

if __name__ == '__main__':
    app.run(debug=True)