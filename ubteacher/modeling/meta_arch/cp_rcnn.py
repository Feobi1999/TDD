# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from detectron2.modeling.postprocessing import detector_postprocess
import numpy as np
import torch
@META_ARCH_REGISTRY.register()
class Ori_TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):




    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):

        import pdb
        pdb.set_trace()
        if (not self.training) and (not val_mode):



            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)





        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None



    def inference(
        self,
        batched_inputs,
        detected_instances= None,
        do_postprocess= True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        else:
            gt_instances = None

        assert not self.training
        import pdb
        # pdb.set_trace()
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        features_save=[]

        img_id = [x["image_id"] for x in batched_inputs]

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            # proposals = [x["instances"].to(self.device) for x in batched_inputs]

            import pdb

            results, box_features_save = self.roi_heads(images, features, proposals, None)
            #     results,box_features_save = self.roi_heads(images, features, proposals, None, branch='proposal_pooling')
            # else:
            #     results,box_features_save = self.roi_heads(images, features, gt_instances, None,branch="gt_pooling")
            # pred_cls = results[0]._fields['pred_classes']
            pred_cls = results[0]._fields['pred_classes']
            pred_scores = results[0]._fields['scores']
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results  = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            # return self._postprocess(results, batched_inputs, images.image_sizes)
            # return self._postprocess(results, batched_inputs, images.image_sizes), box_features_save, pred_cls ,pred_scores
            return self._postprocess(results, batched_inputs, images.image_sizes), features_save

            # return self._postprocess(results, batched_inputs, images.image_sizes), box_features_save, pred_cls, img_id, proposals
        else:
            return results

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            import pdb
            # pdb.set_trace()
            processed_results.append({"instances": r})

        return processed_results