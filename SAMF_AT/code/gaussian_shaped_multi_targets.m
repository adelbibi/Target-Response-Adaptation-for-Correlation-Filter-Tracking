function resultant_max = gaussian_shaped_multi_targets(sigma, y_o, sz)
%GAUSSIAN_SHAPED_LABELS
%   Gaussian-shaped labels for all shifts of a sample.
%
%   LABELS = GAUSSIAN_SHAPED_LABELS(SIGMA, SZ)
%   Creates an array of labels (regression targets) for all shifts of a
%   sample of dimensions SZ. The output will have size SZ, representing
%   one label for each possible shift. The labels will be Gaussian-shaped,
%   with the peak at 0-shift (top-left element of the array), decaying
%   as the distance increases, and wrapping around at the borders.
%   The Gaussian function has spatial bandwidth SIGMA.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


% 	%as a simple example, the limit sigma = 0 would be a Dirac delta,
% 	%instead of a Gaussian:
% 	labels = zeros(sz(1:2));  %labels for all shifted samples
% 	labels(1,1) = magnitude;  %label for 0-shift (original sample)

[r_targets, c_targets] = find(y_o~=0);
locations_2B_shifted = [r_targets, c_targets];

[cs, rs] = meshgrid(1:sz(2),1:sz(1));
cs = cs  - floor(sz(2)/2) ;
rs = rs  - floor(sz(1)/2) ;

%move the peak to the top-left, with wrap-around
% 	labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
for i=1:size(locations_2B_shifted,1)
    labels(:,:,i) = y_o(r_targets(i), c_targets(i))*circshift(exp(-0.5 / sigma^2 * (rs.^2 + cs.^2)), floor(locations_2B_shifted(i,:)- floor(sz/2)));
end
resultant_max = max(labels,[],3);

end

