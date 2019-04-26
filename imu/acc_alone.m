% %
% % acc = vals(1:3,:);
% % theta = vals(4:6,:);
% dt = [data_imu.ts,0] - [0,data_imu.ts];
% dt = dt(2:end-1);
% % %
% % % R = eye(3);
% % % omega = [0;0;0];
% % size_t = size(acc,2);
% % % for t = 1:size_t
% % %     omega = omega + dt(t)*acc(t);
% % %     omega_hat =[0 -omega(3) omega(2) ; omega(3) 0 -omega(1) ; -omega(2) omega(1) 0 ];
% % %     R = R + dt(t) * omega_hat;
% % %     norm(R - rots(t),'fro')
% % % %     norm(R,'fro') - norm(rots(t),'fro')
% % % end
% %
% %
% % %theta
% R = eye(3);
% % omega = [0;0;0];
% 
% % d_theta = [omega, theta] - [theta, omega];
% % d_theta = d_theta(:,2:end-1);
% % diff = [];
% quat = [1, 0, 0, 0];
% rpy = [];
% sens = 3;
% sf = 3300/1023/sens*pi/180;
% omega_data = [];
% 
% for t = 1:size_t
%     omega = ([gyro(2,t); gyro(3,t); gyro(1,t)] - [  371.3300  374.7600  376.2000]')*sf;
% %     quat_gyro = [0, omega'];
% %     q_grad = 0.5*quatmultiply(quat,quat_gyro);
%     axis = omega/norm(omega);
%     angle = norm(omega)*dt(t);
%     delta_quat = [cos(angle/2), axis'.*sin(angle/2)];
% %     delta_quat = [1, 0.5*dt(t)*omega'];
%     quat = quatmultiply(quat, delta_quat);
% %             omega_skew = [0 -omega(3) omega(2) ; omega(3) 0 -omega(1) ; -omega(2) omega(1) 0 ];
% %             R = R + dt(t)*omega_skew*R;
%     
%     %     quat = quat + q_grad*dt(t);
%     rpy = [rpy; quat2eul(quat, 'ZYX')];
% end
acc = [];
for i = quat'
v = quatmultiply(quatmultiply(i', [0,0,0,9.8]), quatinv(i'));
acc = [acc; v(2:end)];
end

ts = data_vicon.ts;
dt = [ts,0] - [0,ts];
dt = dt(2:end-1);
size_t = size(ts,2);
omegas = [];
for t = 2:size_t
    R_dot = (data_vicon.rots(:,:,t) - data_vicon.rots(:,:,t-1))/dt(t-1);
    w = inv(data_vicon.rots(:,:,t))*R_dot;
     omegas = [omegas; -w(1,2), -w(2,3), w(1,3)];
end
omegas

