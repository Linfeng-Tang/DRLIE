from ops import *
from utils import *
from glob import glob
import time
import h5py
import numpy as np
import os


class DRIT(object):
    def __init__(self, sess, args):
        self.model_name = 'DRIT'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.test_dir = args.test_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.decay_epoch = args.decay_epoch

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.num_attribute = args.num_attribute  # for test
        self.guide_img = args.guide_img
        self.direction = args.direction

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.init_lr = args.lr
        self.content_init_lr = args.lr * 5
        self.ch = args.ch
        self.concat = args.concat

        """ Weight """
        self.content_adv_w = args.content_adv_w
        self.domain_adv_w = args.domain_adv_w
        self.fake_w = args.fake_w
        self.recon_w = args.recon_w
        self.att_w = args.att_w
        self.kl_w = args.kl_w

        """ Generator """
        self.n_layer = args.n_layer
        self.n_z = args.n_z

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_scale = args.n_scale
        self.n_d_con = args.n_d_con
        self.multi = True if args.n_scale > 1 else False
        self.sn = args.sn
        self.guide_num = args.guide_num
        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

    def content_encoder(self, x, is_training=True, reuse=False, scope='content_encoder'):
        feature_map = []
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv')
            x = lrelu(x, 0.01)
            for i in range(2):
                feature_map.append(x) 
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                # x = layer_norm(x, scope='layer_norm_' + str(i))
                x = relu(x)

                channel = channel * 2

            for i in range(0, self.n_layer):
                x = resblock(x, channel, scope='resblock_' + str(i))
            x = gaussian_noise_layer(x, is_training)
        return x, feature_map

    def attribute_encoder(self, x, reuse=False, scope='attribute_encoder'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv')
            x = relu(x)
            channel = channel * 2

            x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_0')
            x = relu(x)
            channel = channel * 2

            for i in range(1, self.n_layer):
                x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv_' + str(i))
                x = relu(x)

            x = global_avg_pooling(x)
            x = conv(x, channels=self.n_z, kernel=1, stride=1, scope='attribute_logit')

            return x

    def attribute_encoder_concat(self, x, reuse=False, scope='attribute_encoder_concat'):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=4, stride=2, pad=1, pad_type='reflect', scope='conv')

            for i in range(1, self.n_layer):
                channel = channel * (i + 1)
                x = basic_block(x, channel, scope='basic_block_' + str(i))

            x = lrelu(x, 0.2)
            x = global_avg_pooling(x)

            mean = fully_conneted(x, channels=self.n_z, scope='z_mean')
            logvar = fully_conneted(x, channels=self.n_z, scope='z_logvar')
            print('concat attribute encoder!')

            return mean, logvar

    def MLP(self, z, reuse=False, scope='MLP'):
        channel = self.ch * self.n_layer
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(2):
                z = fully_conneted(z, channel, scope='fully_' + str(i))
                z = relu(z)

            z = fully_conneted(z, channel * self.n_layer, scope='fully_logit')

            return z

    def generator(self, x, z, f=None,reuse=False, scope="generator"):
        channel = self.ch * self.n_layer * 2
        with tf.variable_scope(scope, reuse=reuse):
            z = self.MLP(z, reuse=reuse)
            z = tf.split(z, num_or_size_splits=self.n_layer, axis=-1)
            for i in range(self.n_layer):
                x = mis_resblock(x, z[i], channel, scope='mis_resblock_' + str(i))

            for i in range(2):
                feature = f[i]
                x = deconv(x, feature=feature, channels = channel // 2, kernel=3, stride=2, scope='deconv_' + str(i))
                x = relu(x)
                x = tf.concate([x, feature], -1)
                channel = channel // 2
            x = conv(x, channels=self.img_ch, kernel=3, stride=1, pad=1, pad_type='reflect', scope='conv')
            x = lrelu(x)
            x = conv(x, channels=self.img_ch, kernel=1, stride=1, scope='G_logit')
            x = tanh(x)
            return x

    def generator_concat(self, x, z, f=None, reuse=False, scope='generator_concat'):
        channel = self.ch * self.n_layer * 2
        with tf.variable_scope('generator_concat_share', reuse=tf.AUTO_REUSE):
            x = resblock(x, channel, scope='resblock')

        with tf.variable_scope(scope, reuse=reuse):
            channel1 = channel + self.n_z
            x = expand_concat(x, z)

            for i in range(1, self.n_layer):
                x = resblock(x, channel1, scope='resblock_' + str(i))

            x = conv(x, channels=channel, kernel=1, stride=1, scope='change_channel')
            for i in range(2):
                # channel = channel + self.n_z
                feature = f[1 - i]
                x = expand_concat(x, z)
                x = deconv(x, feature=feature, channels=channel // 2, kernel=4, stride=2, scope='deconv_' + str(i))
                # x = layer_norm(x, scope='layer_norm_' + str(i))
                x = relu(x)
                x = x + feature#tf.concat([x, feature], -1)

                channel = channel // 2
            channel = x.get_shape().as_list()[-1]
            x = expand_concat(x, z)
            channel = channel + self.n_z
            for i in range(self.n_layer):
                x = resblock(x, channel, scope='resblock_' + str(i + 2))
            x = conv(x, channels=self.img_ch, kernel=1, stride=1, scope='G_logit')
            x = tanh(x)

            return x

    ##################################################################################
    # Discriminator
    ##################################################################################


    def multi_discriminator(self, x_init, reuse=False, scope="multi_discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse):
            for scale in range(self.n_scale):
                channel = self.ch
                x = conv(x_init, channel, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn,
                         scope='ms_' + str(scale) + 'conv_0')
                x = lrelu(x, 0.01)

                for i in range(1, self.n_dis):
                    x = conv(x, channel * 2, kernel=4, stride=2, pad=1, pad_type='reflect', sn=self.sn,
                             scope='ms_' + str(scale) + 'conv_' + str(i))
                    x = lrelu(x, 0.01)

                    channel = channel * 2

                x = conv(x, channels=1, kernel=1, stride=1, sn=self.sn, scope='ms_' + str(scale) + 'D_logit')
                x = tf.clip_by_value(x, 1e-8, 2.0)
                D_logit.append(x)

                x_init = down_sample(x_init)
            return D_logit

    def discriminator(self, x, reuse=False, scope="discriminator"):
        D_logit = []
        with tf.variable_scope(scope, reuse=reuse):
            channel = self.ch
            x = conv(x, channel, kernel=3, stride=2, pad=1, pad_type='reflect', sn=self.sn, scope='conv')
            x = lrelu(x, 0.01)

            for i in range(1, self.n_dis):
                x = conv(x, channel * 2, kernel=3, stride=2, pad=1, pad_type='reflect', sn=self.sn,
                         scope='conv_' + str(i))
                x = lrelu(x, 0.01)

                channel = channel * 2

            x = conv(x, channels=1, kernel=1, stride=1, sn=self.sn, scope='D_logit')
            x = tf.clip_by_value(x, 1e-8, 2.0)
            print("discriminator x shape:", x.get_shape().as_list())
            D_logit.append(x)

            return D_logit


    ##################################################################################
    # Model
    ##################################################################################

    def Encoder(self, x_A, is_training=True, random_fake=False, reuse=False):
        mean = None
        logvar = None
        content_A, feature_A = self.content_encoder(x_A, is_training=is_training, reuse=reuse, scope='content_encoder_A')

        if self.concat:
            mean, logvar = self.attribute_encoder_concat(x_A, reuse=reuse, scope='attribute_encoder_concat_A')
            if random_fake:
                attribute_A = mean
            else:
                attribute_A = z_sample(mean, logvar)
        else:
            attribute_A = self.attribute_encoder(x_A, reuse=reuse, scope='attribute_encoder_A')

        return content_A, attribute_A, feature_A, mean, logvar

    def Decoder(self, content_B, attribute_A, feature_A, reuse=False):
        if self.concat:
            x = self.generator_concat(x=content_B, z=attribute_A, f=feature_A, reuse=reuse, scope='generator_concat_A')
        else:
            x = self.generator(x=content_B, z=attribute_A, f=feature_A, reuse=reuse, scope='generator_A')

        return x

    def discriminate_real(self, x_A, x_B):
        if self.multi:
            real_A_logit = self.multi_discriminator(x_A, scope='multi_discriminator_A')
            real_B_logit = self.multi_discriminator(x_B, scope='multi_discriminator_B')

        else:
            real_A_logit = self.discriminator(x_A, scope="discriminator_A")
            real_B_logit = self.discriminator(x_B, scope="discriminator_B")

        return real_A_logit, real_B_logit

    def discriminate_fake(self, x_ba, x_ab):
        if self.multi:
            fake_A_logit = self.multi_discriminator(x_ba, reuse=True, scope='multi_discriminator_A')
            fake_B_logit = self.multi_discriminator(x_ab, reuse=True, scope='multi_discriminator_B')

        else:
            fake_A_logit = self.discriminator(x_ba, reuse=True, scope="discriminator_A")
            fake_B_logit = self.discriminator(x_ab, reuse=True, scope="discriminator_B")

        return fake_A_logit, fake_B_logit

    def discriminate_content(self, content_A, content_B, reuse=False):
        content_A_logit = self.content_discriminator(content_A, reuse=reuse, scope='content_discriminator')
        content_B_logit = self.content_discriminator(content_B, reuse=True, scope='content_discriminator')
        return content_A_logit, content_B_logit

    def discriminate_attribute(self, attribute_A, attribute_B, reuse=False):
        attribute_B_logit = self.attribute_discriminator(attribute_B, reuse=reuse, scope='attribute_discriminator')
        attribute_A_logit = self.attribute_discriminator(attribute_A, reuse=True, scope='attribute_discriminator')
        return attribute_A_logit, attribute_B_logit

    def build_model(self):
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.content_lr = tf.placeholder(tf.float32, name='content_lr')

        """ Input Image"""

        self.domain_A = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size, self.img_ch),
                                       name='domain_A')
        self.domain_B = tf.placeholder(tf.float32, shape=(self.batch_size, self.img_size, self.img_size, self.img_ch),
                                       name='domain_B')

        # encode
        with tf.device('/gpu:0'):
            content_a, attribute_a, feature_a, mean_a, logvar_a = self.Encoder(self.domain_A)
            content_b, attribute_b, feature_b, mean_b, logvar_b = self.Encoder(self.domain_B, reuse=True)

        with tf.device('/gpu:1'):
            fake_a = self.Decoder(content_B=content_b, attribute_A=attribute_a, feature_A=feature_b)
            fake_b = self.Decoder(content_B=content_a, attribute_A=attribute_b, feature_A=feature_a, reuse=True)
            recon_a = self.Decoder(content_B=content_a, attribute_A=attribute_a, feature_A=feature_a, reuse=True)
            recon_b = self.Decoder(content_B=content_b, attribute_A=attribute_b, feature_A=feature_b, reuse=True)

        # discriminate
        with tf.device('/gpu:0'):
            real_A_logit, real_B_logit = self.discriminate_real(self.domain_A, self.domain_B)
            fake_A_logit, fake_B_logit = self.discriminate_fake(fake_a, fake_b)

        """ Define Loss """
        g_adv_loss_a = generator_loss(self.gan_type, fake_A_logit)
        g_adv_loss_b = generator_loss(self.gan_type, fake_B_logit)
        g_con_loss = self.content_adv_w * L1_loss(content_a, content_b) + self.content_adv_w * l2_regularize(content_a)\
                     + self.content_adv_w * l2_regularize(content_b)

        self.GC_loss = g_con_loss

        g_rec_loss_a = 2 * L1_loss(recon_a, self.domain_A) + Gradient_loss(recon_a, self.domain_A)
        g_rec_loss_b = 2 * L1_loss(recon_b, self.domain_B) + Gradient_loss(recon_b, self.domain_B)
        # Add a cross-fake loss as it is already paired data
        g_fake_loss_a = 2 * L1_loss(fake_a, self.domain_A) + Gradient_loss(fake_a, self.domain_A)
        g_fake_loss_b = 2 * L1_loss(fake_b, self.domain_B) + Gradient_loss(fake_b, self.domain_B)
        if self.concat:
            g_kl_loss_a = kl_loss(mean_a, logvar_a)
            g_kl_loss_b = kl_loss(mean_b, logvar_b)
        else:
            g_kl_loss_a = l2_regularize(attribute_a)  # + l2_regularize(content_a)
            g_kl_loss_b = l2_regularize(attribute_b)  # + l2_regularize(content_b)
        # Discriminator Loss
        d_adv_loss_a = discriminator_loss(self.gan_type, real_A_logit, fake_A_logit)
        d_adv_loss_b = discriminator_loss(self.gan_type, real_B_logit, fake_B_logit)

        Generator_A_domain_loss = self.domain_adv_w * g_adv_loss_a
        self.GAD_loss = Generator_A_domain_loss
        Generator_A_attribute_loss = self.att_w * g_kl_loss_a
        self.GAA_loss = Generator_A_attribute_loss
        Generator_A_recon_loss = 1 * self.recon_w * g_rec_loss_a
        self.GAR_loss = Generator_A_recon_loss
        Generator_A_fake_loss = self.fake_w * g_fake_loss_a
        self.GAF_loss = Generator_A_fake_loss
        Generator_content_loss = 10 * g_con_loss

        Generator_A_loss = Generator_A_domain_loss + \
                           Generator_A_attribute_loss + \
                           Generator_A_recon_loss + \
                           Generator_A_fake_loss + \
                           Generator_content_loss
        # Generator_A_content_loss + \
        Generator_B_domain_loss = self.domain_adv_w * g_adv_loss_b
        self.GBD_loss = Generator_B_domain_loss

        Generator_B_attribute_loss = 5 * self.att_w * g_kl_loss_b
        self.GBA_loss = Generator_B_attribute_loss
        Generator_B_recon_loss = self.recon_w * g_rec_loss_b
        self.GBR_loss = Generator_B_recon_loss
        Generator_B_fake_loss = self.fake_w * g_fake_loss_b
        self.GBF_loss = Generator_B_fake_loss
        Generator_B_perceptual_loss = 5 * g_fake_loss_b
        # self.GCon_loss = self.GACon_loss + self.GBCon_loss
        self.GA_loss = self.GAA_loss + self.GBA_loss
        Generator_B_loss = Generator_B_domain_loss + \
                           Generator_B_attribute_loss + \
                           Generator_B_fake_loss + \
                           Generator_B_recon_loss
                           # Generator_B_content_loss + \

        Discriminator_A_loss = self.domain_adv_w * d_adv_loss_a
        self.DA_loss = Discriminator_A_loss
        Discriminator_B_loss = self.domain_adv_w * d_adv_loss_b
        self.DB_loss = Discriminator_B_loss

        self.Generator_loss = Generator_A_loss + Generator_B_loss + g_con_loss
        self.Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'encoder' in var.name or 'generator' in var.name]
        variables_file = 'generator_variables.txt'
        if os.path.exists(variables_file):
            os.remove(variables_file)
        for var in G_vars:
            with open(variables_file, 'a') as log:
                log.write(var.name)
                log.write('\n')
        D_vars = [var for var in t_vars if 'discriminator' in var.name and 'attribut' not in var.name]
        variables_file = 'Discriminator_variables.txt'
        if os.path.exists(variables_file):
            os.remove(variables_file)
        for var in D_vars:
            with open(variables_file, 'a') as log:
                log.write(var.name)
                log.write('\n')

        grads_G, _ = tf.clip_by_global_norm(tf.gradients(self.Generator_loss, G_vars), clip_norm=5)

        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).apply_gradients(
            zip(grads_G, G_vars))
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss,
                                                                                        var_list=D_vars)
        """ Image """
        self.fake_A = fake_a
        self.fake_B = fake_b
        self.recon_A = recon_a

        self.real_A = self.domain_A
        self.real_B = self.domain_B

    def train(self):
        dataset_name = './train_mef.h5'
        f = h5py.File(dataset_name, 'r')
        sources = f['data'][:]
        print(sources.shape)
        sources = np.transpose(sources, (0, 3, 2, 1))
        print('sources shape: ', sources.shape)
        num_imgs = sources.shape[0]
        # num_imgs = 800
        mod = num_imgs % self.batch_size
        n_batches = int(num_imgs // self.batch_size)
        print('Train images number %d, Batches: %d.\n' % (num_imgs, n_batches))
        self.iteration = n_batches

        if mod > 0:
            print('Train set has been trimmed %d samples...\n' % mod)
            sources = sources[:-mod]
        print("source shape:", sources.shape)

        batch_idxs = n_batches
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / batch_idxs)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        lr = self.init_lr
        content_lr = self.content_init_lr
        for epoch in range(start_epoch, self.epoch):
            np.random.shuffle(sources)
            if self.decay_flag:
                lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (
                        self.epoch - self.decay_epoch)  # linear decay
                content_lr = self.content_init_lr if epoch < self.decay_epoch else self.content_init_lr * (
                        self.epoch - epoch) / (self.epoch - self.decay_epoch)  # linear decay

            for idx in range(0, batch_idxs):
                patch_A = sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 0:3]
                patch_B = sources[idx * self.batch_size:(idx * self.batch_size + self.batch_size), :, :, 3:6]
                patch_A = (patch_A - 0.5) * 2
                patch_B = (patch_B - 0.5) * 2
                train_feed_dict = {
                    self.lr: lr,
                    self.content_lr: content_lr,
                    self.domain_A: patch_A,
                    self.domain_B: patch_B
                }
                # Update D
                _, d_loss = self.sess.run([self.D_optim, self.Discriminator_loss],
                                          feed_dict=train_feed_dict)

                # Update G
                # , summary_str
                batch_A_images, batch_B_images, fake_A, fake_B, recon_A, _, g_loss = self.sess.run(
                    [self.real_A, self.real_B, self.fake_A, self.fake_B, self.recon_A, self.G_optim,
                     self.Generator_loss], feed_dict=train_feed_dict)

                if idx % 100 == 0:
                    GAD_loss, GAF_loss, GAR_loss, GAA_loss = self.sess.run(
                        [self.GAD_loss, self.GAF_loss, self.GAR_loss, self.GAA_loss],
                        feed_dict=train_feed_dict)
                    print("The loss of domain A:")
                    print("Epoch: [%2d/%2d] [%4d/%4d], Domain loss:[%.4f], "
                          "attribute loss:[%.4f], Reconstruct loss:[%.4f], Fake loss:[%.4f]"
                          % (
                              epoch, self.epoch, idx, batch_idxs, GAD_loss, GAA_loss, GAR_loss, GAF_loss))
                    
                    GBD_loss, GCon_loss, GBA_loss, GBR_loss, GBF_loss = self.sess.run(
                        [self.GBD_loss, self.GC_loss, self.GBA_loss, self.GBR_loss, self.GBF_loss],
                        feed_dict=train_feed_dict)
                    print("The loss of domain B:")
                    print("Epoch: [%2d/%2d] [%4d/%4d], Domain loss:[%.4f], Generator_Content loss: [%.4f], "
                          "attribute loss:[%.4f], Reconstruct loss:[%.4f], Fake loss:[%.4f]"
                          % (epoch, self.epoch, idx, batch_idxs, GBD_loss, GCon_loss, GBA_loss, GBR_loss, GBF_loss))
                    
                    DA_loss, DB_loss= self.sess.run(
                        [self.DA_loss, self.DB_loss],
                        feed_dict=train_feed_dict)
                    print("The loss of discriminator:")
                    print(
                        "Epoch: [%2d/%2d] [%4d/%4d], Discriminator_A loss:[%.4f], Discriminator_B loss: [%.4f]"
                        % (epoch, self.epoch, idx, batch_idxs, DA_loss, DB_loss))
                    
                if np.mod(idx + 1, self.print_freq) == 0:
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))
                    save_images(batch_B_images, [self.batch_size, 1],
                                './{}/real_B_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))
                    save_images(fake_A, [self.batch_size, 1],
                                './{}/fake_A_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))
                    save_images(recon_A, [self.batch_size, 1],
                                './{}/recon_A_{:03d}_{:05d}.jpg'.format(self.sample_dir, epoch, idx + 1))

                # display training status
                counter += 1

                if np.mod(idx + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, counter)

            self.save(self.checkpoint_dir, counter)

    @property
    def model_dir(self):
        if self.concat:
            concat = "_concat"
        else:
            concat = ""
        if self.sn:
            sn = "_sn"
        else:
            sn = ""
        return "{}{}_{}_{}layer_{}dis_{}scale{}".format(self.model_name, concat,
                                                                 self.gan_type,
                                                                 self.n_layer, self.n_dis, self.n_scale,
                                                                 sn)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def guide_test(self):
        if self.direction == 'a2b':
            tf.global_variables_initializer().run()
            test_dir = r'./dataset/{}'.format(self.dataset_name)
            self.saver = tf.train.Saver()
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            if could_load:
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            """ Guided Image Translation """

            filelist = os.listdir(test_dir)
            filelist.sort(key=lambda x: int(x[0:-4]))
            guide_file = r'guide/'+ str(self.guide_num) + '.png'
            guide_name = os.path.basename(guide_file)
            self.save_dir = os.path.join(self.result_dir, self.dataset_name) 
            check_folder(self.save_dir)            
            self.content_image_A = tf.placeholder(tf.float32, [1, None, None, self.img_ch],
                                                      name='content_image_A')
            self.attribute_image = tf.placeholder(tf.float32, [1, None, None, self.img_ch],
                                                    name='guide_attribute_image')

            if self.direction == 'a2b':
                with tf.device('/gpu:1'):
                    content_A, feature_A = self.content_encoder(self.content_image_A, is_training=False,
                                                        reuse=True, scope='content_encoder_A')
                with tf.device('/gpu:1'):
                    guide_mean, guide_logvar = self.attribute_encoder_concat(self.attribute_image, reuse=True,
                                                                                scope='attribute_encoder_concat_A')
                    guide_attribute = z_sample(guide_mean, guide_logvar)
                with tf.device('/gpu:1'):
                    self.fusion = self.Decoder(content_A, guide_attribute, feature_A,
                                                reuse=True)
                self.content_A = content_A

            for item in filelist:
                sample_file = os.path.join(os.path.abspath(test_dir), item)
                sample_image, h, w = load_test_data(sample_file, size=self.img_size)
                sample_image = np.asarray(sample_image)
                save_path = os.path.join(self.save_dir, '{}'.format(os.path.basename(sample_file) ))
                print(save_path)
                attribute_file, h1, w1 = load_test_data(guide_file, size=self.img_size)
                attribute_file = np.asarray(attribute_file)
                fusion_image, content_A = self.sess.run(
                    [self.fusion, self.content_A],
                    feed_dict={self.content_image_A: sample_image,
                               self.attribute_image: attribute_file})
                sample_image = (sample_image + 1) / 2
                fusion_image = (fusion_image + 1) / 2
                attention_map = 1 - sample_image
                fusion_image = 2 * fusion_image - 1                
                fusion_image = (fusion_image - np.min(fusion_image)) / (np.max(fusion_image) - np.min(fusion_image))
                save_images(fusion_image, [1, 1], save_path)