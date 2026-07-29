function [ r, drdqdd, drdqd, drdq, c0] = residual_galerkin_rom( q, qd, qdd, t, Assembly, Fext, Phi, M_red, C_red)
%  RESIDUAL_GALERKIN_ROM Residual for time integration of the Galerkin-projected
%  reduced-order model obtained from modal truncation onto the slowest
%  eigenmodes (basis Phi), used for direct comparison against the SSM/GSS
%  reduced dynamics.
%
%  The full-order (constrained) equations of motion are
%
%     M_red*qdd_full + C_red*qd_full + F(q_full) = Fext(t)
%
%  Substituting the Galerkin ansatz q_full = Phi*q (Phi: n_red x m, m = #modes
%  kept, mass-normalized so Phi.'*M_red*Phi = I) and projecting with Phi.'
%  gives the reduced residual
%
%     r(q,qd,qdd) = Phi.'*M_red*Phi*qdd + Phi.'*C_red*Phi*qd
%                   + Phi.'*F(Phi*q) - Phi.'*Fext(t)
%
%  Nonlinear internal force F(.) is still evaluated at full DOF resolution
%  (via Assembly.tangent_stiffness_and_force), then projected down, since the
%  nonlinearity does not commute with the projection.
%
%  Inputs q, qd, qdd are reduced (modal) coordinates of size m x 1.
%
%  M_red, C_red are passed in already projected (Phi.'*M*Phi, Phi.'*C*Phi) to
%  avoid recomputing them at every Newmark iteration.

% Expand reduced state to full constrained DOFs to evaluate the nonlinearity
q_full = Phi * q;
u = Assembly.unconstrain_vector(q_full);
[K, F] = Assembly.tangent_stiffness_and_force(u);
F_elastic_full = Assembly.constrain_vector(F);

% Project nonlinear elastic force back onto the modal basis
F_elastic = Phi.' * F_elastic_full;

F_external = Phi.' * Fext(t).';

F_inertial = M_red * qdd;
F_damping  = C_red * qd;

r = F_inertial + F_damping + F_elastic + F_external;

drdqdd = M_red;
drdqd  = C_red;
% Tangent stiffness projected and re-linearized about the current modal
% state; since K is only used by the Newmark/Newton iteration as a tangent,
% we project it through Phi as well.
K_red_full = Assembly.constrain_matrix(K);
drdq = Phi.' * K_red_full * Phi;

c0 = norm(F_inertial) + norm(F_damping) + norm(F_elastic) + norm(F_external);

end
