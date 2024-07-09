"""
Initializes GAT trainer and trains the model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import tensorflow as tf
from tqdm import tqdm

from src.DataLoader import GetGATDataset
from src.models.GraphAttentionModel import TransGAT
from src.utils.metrics import LossLayer
from src.utils.model_utils import CustomSchedule, _set_up_dirs
from src.utils.rogue import rouge_n
from src.utils.model_utils import process_results

class TrackableOptimizer(tf.Module):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __call__(self):
        return self.optimizer

def _train_gat_trans(args):
    # set up dirs
    (OUTPUT_DIR, EvalResultsFile,
     TestResults, log_file, log_dir) = _set_up_dirs(args)

    # Load the eval src and tgt files for evaluation
    reference = open(args.eval_ref, 'r')
    eval_file = open(args.eval, 'r')

    OUTPUT_DIR += '/{}_{}'.format(args.enc_type, args.dec_type)

    (dataset, eval_set, test_set, BUFFER_SIZE, BATCH_SIZE, steps_per_epoch,
     src_vocab_size, src_vocab, tgt_vocab_size, tgt_vocab, max_length_targ, dataset_size) = GetGATDataset(args)

    # Initialize model and loss
    model = TransGAT(args, src_vocab_size, src_vocab, tgt_vocab, tgt_vocab_size, max_length_targ)
    loss_layer = LossLayer(tgt_vocab_size, 0.1)

    # Initialize optimizer
    if args.decay is not None:
        learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
    else:
        learning_rate = args.learning_rate

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    trackable_optimizer = TrackableOptimizer(optimizer)

    # Save model parameters for future use
    if os.path.isfile('{}/{}_{}_params'.format(log_dir, args.lang, args.model)):
        with open('{}/{}_{}_params'.format(log_dir, args.lang, args.model), 'rb') as fp:
            PARAMS = pickle.load(fp)
            print('Loaded Parameters..')
    else:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        PARAMS = {
            "args": args,
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "max_tgt_length": max_length_targ,
            "dataset_size": dataset_size,
            "step": 0
        }

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # Define checkpoint manager
    ckpt = tf.train.Checkpoint(model=model, optimizer=trackable_optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

    if args.epochs is not None:
        steps = args.epochs * steps_per_epoch
    else:
        steps = args.steps

    def train_step(nodes, labels, node1, node2, targ):
        with tf.GradientTape() as tape:
            predictions = model(nodes=nodes, labels=labels, node1=node1, node2=node2, targ=targ, mask=None)
            predictions = model.metric_layer([predictions, targ])
            batch_loss = loss_layer([predictions, targ])

        gradients = tape.gradient(batch_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc = model.metrics[0].result()
        ppl = model.metrics[-1].result()
        train_loss(batch_loss)  # Update the train_loss metric

        return batch_loss, acc, ppl

    def eval_step(steps=None):
        model.trainable = False
        results = []
        ref_target = []
        eval_results = open(EvalResultsFile, 'w+')
        eval_set_iter = eval_set if steps is None else eval_set.take(steps)

        for batch, (nodes, labels, node1, node2, targets) in tqdm(enumerate(eval_set_iter)):
            predictions = model(nodes, labels, node1, node2, targ=None, mask=None)
            pred = predictions['outputs'].numpy().tolist()

            if args.sentencepiece == 'True':
                for i in range(len(pred)):
                    sentence = tgt_vocab.DecodeIds(list(pred[i]))
                    sentence = sentence.partition("<start>")[2].partition("<end>")[0]
                    eval_results.write(sentence + '\n')
                    ref_target.append(reference.readline())
                    results.append(sentence)
            else:
                for i in pred:
                    sentences = tgt_vocab.sequences_to_texts([i])
                    sentence = [j.partition("<start>")[2].partition("<end>")[0] for j in sentences]
                    eval_results.write(sentence[0] + '\n')
                    ref_target.append(reference.readline())
                    results.append(sentence[0])

        rogue = rouge_n(results, ref_target)
        eval_results.close()
        model.trainable = True

        return rogue

    def test_step():
        model.trainable = False
        results = []
        ref_target = []
        eval_results = open(TestResults, 'w+')

        for batch, (nodes, labels, node1, node2) in tqdm(enumerate(test_set)):
            predictions = model(nodes, labels, node1, node2, targ=None, mask=None)
            pred = predictions['outputs'].numpy().tolist()

            if args.sentencepiece == 'True':
                for i in range(len(pred)):
                    sentence = tgt_vocab.DecodeIds(list(pred[i]))
                    sentence = sentence.partition("<start>")[2].partition("<end>")[0]
                    eval_results.write(sentence + '\n')
                    ref_target.append(reference.readline())
                    results.append(sentence)
            else:
                for i in pred:
                    sentences = tgt_vocab.sequences_to_texts([i])
                    sentence = [j.partition("<start>")[2].partition("<end>")[0] for j in sentences]
                    eval_results.write(sentence[0] + '\n')
                    ref_target.append(reference.readline())
                    results.append(sentence[0])

        rogue = rouge_n(results, ref_target)
        eval_results.close()
        model.trainable = True
        process_results(TestResults)

        return rogue

    if args.mode == 'train':
        train_loss.reset_state()

        for batch, (nodes, labels, node1, node2, targ) in tqdm(enumerate(dataset.repeat(-1))):
            if PARAMS['step'] < steps:
                start = time.time()
                PARAMS['step'] += 1

                batch_loss, acc, ppl = train_step(nodes, labels, node1, node2, targ)
                if batch % 100 == 0:
                    print('Step {} Learning Rate {:.4f} Train Loss {:.4f} '
                          'Accuracy {:.4f} Perplex {:.4f}'.format(PARAMS['step'],
                                                                  optimizer.learning_rate.numpy(),
                                                                  train_loss.result(),
                                                                  acc.numpy(),
                                                                  ppl.numpy()))
                    print('Time {} \n'.format(time.time() - start))

                # Log the training results
                with open(log_file, 'a') as f:
                    f.write(f"Step {PARAMS['step']} Train Accuracy: {acc.numpy()} "
                            f"Loss: {train_loss.result()} Perplexity: {ppl.numpy()} \n")

                if batch % args.eval_steps == 0:
                    metric_dict = eval_step(5)
                    print('\n' + '---------------------------------------------------------------------' + '\n')
                    print('ROGUE {:.4f}'.format(metric_dict))
                    print('\n' + '---------------------------------------------------------------------' + '\n')

                if batch % args.checkpoint == 0:
                    print("Saving checkpoint \n")
                    ckpt_save_path = ckpt_manager.save()
                    with open(log_dir + '/' + args.lang + '_' + args.model + '_params', 'wb+') as fp:
                        pickle.dump(PARAMS, fp)
            else:
                break

        rogue = test_step()
        print('\n' + '---------------------------------------------------------------------' + '\n')
        print('Rogue {:.4f}'.format(rogue))
        print('\n' + '---------------------------------------------------------------------' + '\n')

    elif args.mode == 'test':
        rogue = test_step()
        print('\n' + '---------------------------------------------------------------------' + '\n')
        print('Rogue {:.4f}'.format(rogue))
        print('\n' + '---------------------------------------------------------------------' + '\n')

    else:
        raise ValueError("Mode must be either 'train' or 'test'")
