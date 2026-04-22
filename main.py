import pandas as pd
import os
import sys
from graph import app
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=== Agentic AI Data Cleaning Tool ===")
    
    # 1. file upload (path input)
    file_path = input("enter the path to your raw CSV file (default: dirty_data.csv): ").strip() or "dirty_data.csv"
    if not os.path.exists(file_path):
        print(f"error: file not found at {file_path}")
        return

    # 2. user context
    user_context = input("what is this data for? (e.g., ML modeling, business reporting, auditing): ").strip()
    if not user_context:
        user_context = "general cleaning and formatting"

    # load initial data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"error reading csv: {e}")
        return

    # initialize state
    config = {"configurable": {"thread_id": "cleaning_session_1"}}
    initial_state = {
        "file_path": file_path,
        "current_df": df,
        "df_history": [df],
        "user_context": user_context,
        "analysis_report": "",
        "cleaning_plan": "",
        "logs": [],
        "is_clean": False,
        "messages": []
    }

    print("\n--- starting agentic analysis ---")
    
    # run until human review
    for event in app.stream(initial_state, config):
        for node_name, output in event.items():
            print(f"\n[node: {node_name}]")
            if "logs" in output:
                for log in output["logs"]:
                    print(f"  - {log}")
            
            if node_name == "analyzer":
                print(f"\nanalysis report:\n{output['analysis_report']}")
            
            if node_name == "planner":
                print(f"\nproposed cleaning plan:\n{output['cleaning_plan']}")

    # human review loop
    state = app.get_state(config)
    if state.next:
        print("\n=== plan approval required ===")
        print(f"proposed plan:\n{state.values['cleaning_plan']}")
        
        choice = input("\napprove the plan? (y)es / (e)dit / (q)uit: ").lower()
        
        if choice == 'q':
            print("aborting.")
            return
        elif choice == 'e':
            new_plan = input("enter your edited plan: ")
            app.update_state(config, {"cleaning_plan": new_plan})
            print("plan updated.")
        
        print("\n--- resuming execution ---")
        # resume execution
        for event in app.stream(None, config):
            for node_name, output in event.items():
                print(f"\n[node: {node_name}]")
                if "logs" in output:
                    for log in output["logs"]:
                        print(f"  - {log}")
                
                if node_name == "validator":
                    print(f"\nvalidation report:\n{output['analysis_report']}")

    # final result
    final_state = app.get_state(config)
    final_df = final_state.values["current_df"]
    
    output_path = "cleaned_data.csv"
    final_df.to_csv(output_path, index=False)
    
    print(f"\n=== cleaning complete ===")
    print(f"final data saved to: {output_path}")
    print(f"total steps taken: {len(final_state.values['logs'])}")
    
    # logs export
    with open("cleaning_logs.txt", "w") as f:
        f.write(f"Agentic Cleaning Logs for {file_path}\n")
        f.write(f"Goal: {user_context}\n")
        f.write("-" * 30 + "\n")
        for log in final_state.values["logs"]:
            f.write(f"- {log}\n")
    
    print("logs exported to: cleaning_logs.txt")

if __name__ == "__main__":
    main()
