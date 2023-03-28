close all, clear, clc;

fname = "quadruped_quat_test.json";
fid = fopen(fname);
raw = fread(fid, inf);
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);

reference_state = val.reference_state;
reference_r = reference_state(:, 1:3);
reference_q = reference_state(:, 4:7);
reference_v = reference_state(:, 8:10);
reference_w = reference_state(:, 11:13);

state_trajectory = val.state_trajectory;
r_trajectory = state_trajectory(:, 1:3);
q_trajectory = state_trajectory(:, 4:7);
v_trajectory = state_trajectory(:, 8:10);
w_trajectory = state_trajectory(:, 11:13);

plot(reference_w, 'LineWidth', 10);
grid on, hold on;
plot(w_trajectory, 'LineWidth', 5, 'LineStyle', '--');