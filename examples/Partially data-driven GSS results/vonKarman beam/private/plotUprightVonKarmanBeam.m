% MATLAB code to plot a 3D upright cantilevered von Kármán beam
function h=plotUprightVonKarmanBeam(l, b, h, coordinates, numPoints,colu,setu)
    % Inputs:
    % l - Length of the beam (along z-axis)
    % b - Breadth of the beam (width in x-direction)
    % h - Height of the beam (width in y-direction)
    % coordinates - Matrix of [u, w, theta] (axial disp in z, transverse disp in x, angle about y) at discrete points
    % numPoints - Number of points along the beam for plotting

    % Generate axial coordinates (z) along the beam
    z = linspace(0, l, numPoints);
    
    % Extract displacements and rotation (assuming coordinates is numPoints x 3)
    u = coordinates(:, 1); % Axial displacement (along z)
    w = coordinates(:, 2); % Transverse displacement (in x-direction)
    theta = coordinates(:, 3); % Rotation angle (radians, about y-axis)
    
    % Initialize arrays for 3D coordinates of beam cross-section
    % Beam cross-section is a rectangle with breadth b (x) and height h (y)
    numCrossSectionPoints = 4; % Four corners of rectangular cross-section
    crossSectionX = [-b/2, b/2, b/2, -b/2]; % x-coordinates of cross-section
    
    
    crossSectionY = [-h/2, -h/2, h/2, h/2]; % y-coordinates of cross-section
    
    % Initialize figure
%     figure;
    hold on;
    grid on;
     axis equal;
      set(gcf, 'Renderer', 'opengl'); % Force OpenGL renderer
    % Plot the deformed beam
    for i = 1:numPoints
        % Center of cross-section at current point
        centerX = w(i); % Transverse displacement (x-direction)
        centerZ = z(i) + u(i); % Axial position with displacement (z-direction)
        centerY = 0; % Centered in y-direction
        
        % Rotate cross-section about y-axis based on theta
        R = [cos(theta(i)), sin(theta(i)); -sin(theta(i)), cos(theta(i))]; % Rotation matrix in xz-plane
        crossSectionPoints = zeros(numCrossSectionPoints, 2); % [x, z] after rotation
        
        % Compute rotated cross-section coordinates
        for j = 1:numCrossSectionPoints
            localCoords = R * [crossSectionX(j); 0]; % Rotate x, z=0 in local frame
            crossSectionPoints(j, 1) = localCoords(1); % x-coordinate
            crossSectionPoints(j, 2) = localCoords(2); % z-coordinate
        end
        
        % Global coordinates of cross-section
        globalX = centerX + crossSectionPoints(:, 1);
        globalY = centerY + crossSectionY;
        globalZ = centerZ + crossSectionPoints(:, 2);
        
        % Plot cross-section as a filled patch
        if i == 1
%            globalX = centerX + 2*crossSectionPoints(:, 1);
%         globalY = centerY + crossSectionY;
%         globalZ = centerZ + crossSectionPoints(:, 2);
%         
        
            % At fixed end (z=0), color red
             masi = 1;
%              fill3(globalX, globalY, globalZ, [0.6 0.3 0], 'FaceAlpha', 1);
        elseif i == numPoints
            globalX = centerX + crossSectionPoints(:, 1);
        globalY = centerY + crossSectionY;
        globalZ = centerZ + crossSectionPoints(:, 2);
        
            % At free end, color blue
%             fill3(globalX, globalY, globalZ, colu, 'FaceAlpha', setu);
        else
            % Intermediate sections, color green
             fill3(globalX, globalY, globalZ, colu, 'FaceAlpha', setu ,'EdgeColor', 'none');
        end
        
        if i == 1 || i == numPoints-1
            plot3(globalX([1 2 3 4 1]), globalY([1 2 3 4 1]), globalZ([1 2 3 4 1]), ...
          'k-', 'LineWidth', 1.5); % Outline the cross-section
        end

        % Connect cross-sections for visualization
        if i < numPoints-1
            nextCenterX = w(i+1);
            nextCenterZ = z(i+1) + u(i+1);
            nextCrossSectionPoints = zeros(numCrossSectionPoints, 2);
            for j = 1:numCrossSectionPoints
                localCoords = [cos(theta(i+1)), sin(theta(i+1)); -sin(theta(i+1)), cos(theta(i+1))] * [crossSectionX(j); 0];
                nextCrossSectionPoints(j, 1) = localCoords(1);
                nextCrossSectionPoints(j, 2) = localCoords(2);
            end
            nextGlobalX = nextCenterX + nextCrossSectionPoints(:, 1);
            nextGlobalY = centerY + crossSectionY;
            nextGlobalZ = nextCenterZ + nextCrossSectionPoints(:, 2);
            
            % Plot surface between cross-sections
            for j = 1:numCrossSectionPoints
                jj = mod(j, numCrossSectionPoints) + 1;
                X = [globalX(j), nextGlobalX(j); globalX(jj), nextGlobalX(jj)];
                Y = [globalY(j), nextGlobalY(j); globalY(jj), nextGlobalY(jj)];
                Z = [globalZ(j), nextGlobalZ(j); globalZ(jj), nextGlobalZ(jj)];
                h=surf(X, Y, Z, 'FaceColor', colu, 'FaceAlpha', setu, 'EdgeColor', 'none','EdgeAlpha',setu);
%                 shading flat
                 % Add black edges to longitudinal sides only (between cross-sections)
        % Edge from (globalX(j), globalY(j), globalZ(j)) to (nextGlobalX(j), nextGlobalY(j), nextGlobalZ(j))
        plot3([globalX(j), nextGlobalX(j)], [globalY(j), nextGlobalY(j)], [globalZ(j), nextGlobalZ(j)], ...
              '-', 'LineWidth', 1.5,'color',[0 0 0 setu]);
        % Edge from (globalX(jj), globalY(jj), globalZ(jj)) to (nextGlobalX(jj), nextGlobalY(jj), nextGlobalZ(jj))
        plot3([globalX(jj), nextGlobalX(jj)], [globalY(jj), nextGlobalY(jj)], [globalZ(jj), nextGlobalZ(jj)], ...
              '-', 'LineWidth', 1.5,'color',[0 0 0 setu]);
            end
        end
    end
   
    % Plot centerline for reference
%     centerlineX = w;
%     centerlineY = zeros(size(z));
%     centerlineZ = z + u;
%     plot3(centerlineX, centerlineY, centerlineZ, 'k-', 'LineWidth', 2);
%     
    % Set labels and title
%     xlabel('X (Transverse)');
%     ylabel('Y (Height)');
%     zlabel('Z (Axial)');
%     title('Deformed Upright Cantilevered von Kármán Beam');
    lightangle(-65,30)
     lighting gouraud
%     shading interp; % Flat shading to avoid interpolation warnings
%     light('Position', [1, 1, 1], 'Style', 'infinite'); % Single light source
%     material dull; % Matte material to reduce specularity issues
%     view(45, 30); % Angled 3D view
%     camproj('perspective'); % Realistic depth
 axis equal; % Equal scaling for accurate proportions
grid on; % Add 3D grid
% xlabel('X'); ylabel('Y'); zlabel('Z'); % Label axes
% title('3D Cuboid Surface Plot'); % Add title
rotate3d on; % Enable interactive rotation

% hold off;
    % Set view for 3D visualization
     view(-35,13);
%     hold off;
end

% Example usage
