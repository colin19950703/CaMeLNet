import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable

class TripletMarginLoss(object):
    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='sum')
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction='sum')

    def __call__(self, dist_ap, dist_an):
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss

class CaMLoss(nn.Module):
    """CaM loss.

        Args:
            num_classes (int): number of classes.
            feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes, feat_dim, use_gpu=True):
        super(CaMLoss, self).__init__()

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.batch_size = None
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.zeros(self.num_classes, self.feat_dim).to(self.device)
        self.centers_value = torch.zeros(self.num_classes, self.feat_dim).to(self.device)
        self.centers_cnt = torch.zeros(self.num_classes, self.feat_dim).to(self.device)

    def _Cup(self, embeddings, centers_value, centers_cnt, labels):
        """Centroid Update module.

            Args:
                embeddings (torch.Tensor): Current batch's embedding tensor (batch_size x feat_dim).
                centers_value (torch.Tensor): Cumulative value tensor of the current centroids (num_classes x feat_dim).
                centers_cnt (torch.Tensor): Cumulative tensor of the number of samples for each class (num_classes x 1).
                labels (torch.Tensor): Tensor containing class labels for each sample (batch_size x 1).

            Returns:
                updated_centers (torch.Tensor): Updated centroids tensor (num_classes x feat_dim).
                updated_centers_value (torch.Tensor): Updated cumulative value tensor of the centroids (num_classes x feat_dim).
                updated_centers_cnt (torch.Tensor): Updated cumulative tensor of the number of samples for each class (num_classes x 1).
            """
        updated_centers = torch.zeros(self.num_classes, self.feat_dim).to(self.device)
        updated_centers_value = centers_value.detach().clone()
        updated_centers_cnt = centers_cnt.detach().clone()
        embeddings_clone = embeddings.detach().clone()
        for i in torch.unique(labels):
            updated_centers_value[i] = updated_centers_value[i] + embeddings_clone[labels.eq(i)].sum(dim=0)
            updated_centers_cnt[i] = updated_centers_cnt[i] + labels.eq(i).sum()
            if updated_centers_cnt[i].mean() == 0:
                continue
            updated_centers[i] = (updated_centers_value[i] / updated_centers_cnt[i])
        return updated_centers, updated_centers_value, updated_centers_cnt

    def _normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
            Args:
                x: pytorch Variable
            Returns:
                x: pytorch Variable, same shape as input
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    def _euclidean_dist(self, x, y):
        """
            Args:
                x: pytorch Variable, with shape [m, d]
                y: pytorch Variable, with shape [n, d]
            Returns:
                dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt() # for numerical stability

        return dist

    def _get_anchor_positive_triplet_mask(self, labels, device):
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
            Args:
                labels: tf.int32 `Tensor` with shape [batch_size]
            Returns:
                mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct
        indices_equal = torch.eye(labels.size()[0]).bool().to(device)

        indices_not_equal = ~indices_equal  # flip booleans

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = (torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1))
        # Combine the two masks
        mask = indices_not_equal & labels_equal

        return mask

    def _get_anchor_negative_triplet_mask(self, labels, device):
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
            Args:
                labels: tf.int32 `Tensor` with shape [batch_size]
            Returns:
                mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)

        mask = ~labels_equal  # invert the boolean tensor

        return mask

    def _get_anchor_center_triplet_mask(self, labels):
        # Check that i and j are distinct
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(dim=-1)
        elif len(labels.shape) > 2:
            print('labels dim is wrong!')
        classes = torch.arange(self.num_classes).long().to(self.device)
        labels = labels.expand(self.batch_size, self.num_classes).to(self.device)
        mask = torch.eq(labels, classes.expand(self.batch_size, self.num_classes))
        positive_mask = mask
        negative_mask = torch.logical_not(mask)
        return positive_mask, negative_mask

    def forward(self, embeddings, labels, margin, init_center=False):
        """
        Build the camel loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
            Args:
                labels: labels of the batch, of size (batch_size,)
                embeddings: tensor of shape (batch_size, embed_dim)
                margin: margin for triplet loss
                init_center = Flag to init centroids
            Returns:
                triplet_loss: scalar tensor containing the triplet loss
        """
        self.margin = margin
        self.loss_function = TripletMarginLoss(margin=self.margin)
        self.batch_size = embeddings.size(0)

        # Update centroids
        self.centers, self.centers_value, self.centers_cnt = self._Cup(embeddings, self.centers_value, self.centers_cnt, labels)

        # Init centroids
        if init_center:
            self.centers_value = torch.zeros(self.num_classes, self.feat_dim).to(self.device)
            self.centers_cnt = torch.zeros(self.num_classes, self.feat_dim).to(self.device)

        # Get the pairwise distance matrix
        centers = self._normalize(self.centers)
        embeddings = self._normalize(embeddings)
        pairwise_dist = self._euclidean_dist(embeddings, embeddings)
        pairwise_dist_c = self._euclidean_dist(embeddings, centers)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels, self.device)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels, self.device)
        mask_anchor_positive_c, mask_anchor_negative_c = self._get_anchor_center_triplet_mask(labels)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        anchor_positive_dist_c = mask_anchor_positive_c * pairwise_dist_c
        hardest_anchor_positive_dist = torch.max(anchor_positive_dist, 1, keepdim=True)[0]  # shape (batch_size, 1)
        hardest_anchor_positive_dist_c = torch.max(anchor_positive_dist_c, 1, keepdim=True)[0]  # shape (batch_size, 1)
        hardest_positive_dist = torch.maximum(hardest_anchor_positive_dist, hardest_anchor_positive_dist_c)

        hardest_positive_dist_tile = hardest_positive_dist.repeat(1, self.batch_size).detach().clone()
        hardest_positive_dist_tile_c = hardest_positive_dist.repeat(1, self.num_classes).detach().clone()

        greater = pairwise_dist > hardest_positive_dist_tile
        greater_c = pairwise_dist_c > hardest_positive_dist_tile_c

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        greater_boundary_mask = mask_anchor_negative & greater
        greater_boundary_mask_c = mask_anchor_negative_c & greater_c

        #negative_outside : smallest D_an where D_an > D_ap.
        negative_outside = pairwise_dist + 1e+6 * ~(greater_boundary_mask)
        negative_outside = torch.min(negative_outside, 1, keepdim=True)[0]
        negative_outside_c = pairwise_dist_c + 1e+6 * ~(greater_boundary_mask_c)
        negative_outside_c = torch.min(negative_outside_c, 1, keepdim=True)[0]

        hardest_negative_outside = torch.minimum(negative_outside, negative_outside_c)

        #negative_inside : largest D_an where D_an > D_ap.
        mask_step = greater_boundary_mask.to(dtype=torch.float32).sum(axis=1)
        mask_step_c = greater_boundary_mask_c.to(dtype=torch.float32).sum(axis=1)
        mask_step = mask_step == 0.0
        mask_step_c = mask_step_c == 0.0
        mask_final = mask_step.repeat(self.batch_size, 1).transpose(0, 1)
        mask_final_c = mask_step_c.repeat(self.num_classes, 1).transpose(0, 1)

        negative_inside = torch.where(mask_final, pairwise_dist, 1e+6)
        negative_inside_c = torch.where(mask_final_c, pairwise_dist_c, 1e+6)
        negative_inside = negative_inside * mask_anchor_negative
        negative_inside_c = negative_inside_c * mask_anchor_negative_c
        negative_inside = torch.max(negative_inside, 1, keepdim=True)[0]
        negative_inside_c = torch.max(negative_inside_c, 1, keepdim=True)[0]

        hardest_negative_inside = torch.maximum(negative_inside, negative_inside_c)
        hardest_negative_dist = torch.minimum(hardest_negative_outside, hardest_negative_inside)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        loss = self.loss_function(hardest_positive_dist, hardest_negative_dist)

        # Get final mean loss
        loss = loss / self.batch_size

        return loss