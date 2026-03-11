import time

def Test(X_test, y_test, model, args):
    all_results = []

    # number of exits = number of split points inside the wrapper
    num_exits = len(model.split_points)
    n_classes = len(set(y_test))
    
    start_factor_state = model.default_start_factor
    
    for w in range(len(X_test)):
        t_start = time.time()

        pred, dbg = model.predict_one(X_test[w:w+1], sample_id=w, start_factor=start_factor_state)

        t_end = time.time()
        start_factor_state = int(model.next_start_factor_policy(dbg, n_classes))

        stages = dbg.get("stages", [])
        exit_level = stages[-1]["stage"] if stages else -1

        row = {
            "t_start": float(t_start),
            "t_end": float(t_end),
            "total": float(t_end - t_start),
            "true_label": int(y_test[w]),
            "prediction": int(pred),
            "correctness": int(int(pred) == int(y_test[w])),
            "exit_level": int(exit_level),
            "window_num": int(w),
            "data%": float(stages[-1]["split_point"] * 100.0) if stages else -1.0,
        }

        # stage timing columns
        for i in range(1, num_exits + 1):
            row[f"t{i}"] = -1.0

        for st in stages:
            s = int(st["stage"])
            if 1 <= s <= num_exits:
                row[f"t{s}"] = float(st.get("stage_time_sec", -1.0))

        all_results.append(row)

    return all_results