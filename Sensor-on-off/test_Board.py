import time


def full_window_time_sec(window_len, fs_base):
    return float(window_len) / float(fs_base)


def stage_acquisition_times(split_points, window_len, fs_base):
  
    T_window = full_window_time_sec(window_len, fs_base)
    prev = 0.0
    out = []
    for p in split_points:
        p = float(p)
        seg_prop = max(0.0, p - prev)
        out.append(seg_prop * T_window)
        prev = p
    return out


def TestBoardControlled(X_test,y_test,model, args,sensor_on,sensor_sleep,fs_base,window_len,sensor_wakeup_sec=0.0,print_trace=True,):
    all_results = []

    num_exits = len(model.split_points)
    n_classes = len(set(y_test))

    T_window = full_window_time_sec(window_len, fs_base)
    acq_times = stage_acquisition_times(model.split_points, window_len, fs_base)

    start_factor_state = model.default_start_factor

    for w in range(len(X_test)):
        if print_trace:
            print("\n" + "=" * 60)
            print(f"Window {w}")

        t_start = time.time()

        x_one = X_test[w:w + 1]
        x_acc = None
        factor_next = None
        H_prev = None
        pred = -1
        exit_level = -1

        executed_stages = []
        sensor_total_on_sec = 0.0
        compute_total_sec = 0.0

        # -------- START OF WINDOW: SENSOR ON --------
        sensor_on()
        if sensor_wakeup_sec > 0:
            time.sleep(sensor_wakeup_sec)
            sensor_total_on_sec += sensor_wakeup_sec

        for k in range(num_exits):
            # keep sensor ON only for the NEW required segment time
            seg_wait = float(acq_times[k])

            if print_trace:
                print(f"  Stage {k+1}: sensor ON for new segment = {seg_wait:.6f} sec")

            time.sleep(seg_wait)
            sensor_total_on_sec += seg_wait

            pred, stage_info, x_acc, factor_next, H_prev, exit_now = model.predict_one_stage(
                x_full_one=x_one,
                stage_idx=k,
                x_acc=x_acc,
                start_factor=start_factor_state,
                factor_next=factor_next,
                H_prev=H_prev,
                sample_id=w,
                print_trace=print_trace,
            )

            compute_sec = float(stage_info.get("stage_time_sec", 0.0))
            compute_total_sec += compute_sec
            executed_stages.append(stage_info)

            if exit_now:
                exit_level = int(k + 1)
                sensor_sleep()
                remaining_off_time = max(0.0, T_window - sensor_total_on_sec)
                time.sleep(remaining_off_time)
                if print_trace:
                    print(f"  -> EXIT at stage {exit_level}")
                break

        # -------- EXIT HAPPENED: SENSOR OFF FOR REMAINING WINDOW TIME --------
        

        # remaining_off_time = max(0.0, T_window - sensor_total_on_sec)

        if print_trace:
            print(f"  Sensor total ON  = {sensor_total_on_sec:.6f} sec")
            print(f"  Sensor total OFF = {remaining_off_time:.6f} sec (remaining window time)")

        

        t_end = time.time()

        # cross-window next start factor
        dbg = {
            "stages": executed_stages,
            "start_factor": start_factor_state,
        }
        start_factor_state = int(model.next_start_factor_policy(dbg, n_classes))

        row = {
            "t_start": float(t_start),
            "t_end": float(t_end),
            "total": float(t_end - t_start),
            "window_sched_sec": float(T_window),
            "sensor_total_on_sec": float(sensor_total_on_sec),
            "sensor_total_off_sec": float(remaining_off_time),
            "compute_total_sec": float(compute_total_sec),
            "true_label": int(y_test[w]),
            "prediction": int(pred),
            "correctness": int(int(pred) == int(y_test[w])),
            "exit_level": int(exit_level),
            "window_num": int(w),
            "data%": float(executed_stages[-1]["split_point"] * 100.0) if executed_stages else -1.0,
        }

        for i in range(1, num_exits + 1):
            row[f"t{i}_acq"] = -1.0
            row[f"t{i}_compute"] = -1.0

        prev_split = 0.0
        for st in executed_stages:
            s = int(st["stage"])
            split_p = float(st["split_point"])
            row[f"t{s}_acq"] = float((split_p - prev_split) * T_window)
            row[f"t{s}_compute"] = float(st.get("stage_time_sec", -1.0))
            prev_split = split_p

        all_results.append(row)

    return all_results
